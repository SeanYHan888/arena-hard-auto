import argparse
import json
import os
import re
import time
import concurrent.futures
import threading

import tiktoken
import shortuuid
import tqdm

from utils.add_markdown_info import count_markdown_elements, remove_pattern
from utils.completion import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    registered_api_completion,
    registered_engine_completion,
    reorg_answer_file,
    API_ERROR_OUTPUT,
)

ROLE_MARKER_RE = re.compile(r"(^|\n)(user|assistant|system)\n", re.IGNORECASE)
DEFAULT_SANITY_MIN_CHARS = 16
DEBUG_DUMP_LOCK = threading.Lock()


def filter_questions(questions: list[dict], config: dict):
    filtered = questions

    categories = config.get("question_categories")
    if categories:
        category_set = set(categories)
        filtered = [
            question for question in filtered
            if question.get("category") in category_set or question.get("subcategory") in category_set
        ]

    question_uids = config.get("question_uids")
    if question_uids:
        question_by_uid = {question["uid"]: question for question in filtered}
        missing_uids = [uid for uid in question_uids if uid not in question_by_uid]
        if missing_uids:
            print(f"Warning: requested question_uids were not found: {missing_uids}")
        filtered = [question_by_uid[uid] for uid in question_uids if uid in question_by_uid]

    question_offset = int(config.get("question_offset", 0) or 0)
    if question_offset:
        filtered = filtered[question_offset:]

    question_limit = config.get("question_limit")
    if question_limit is not None:
        filtered = filtered[: int(question_limit)]

    return filtered


def sanitize_answer_text(answer: str):
    if not isinstance(answer, str):
        return "", ["non_string_answer"]

    cleaned = answer
    cleanup_actions = []
    stripped = cleaned.lstrip()
    leading_ws = len(cleaned) - len(stripped)

    if stripped.lower().startswith("<think>"):
        close_idx = stripped.lower().find("</think>")
        if close_idx != -1:
            stripped = stripped[close_idx + len("</think>") :]
            cleaned = cleaned[:leading_ws] + stripped
            cleanup_actions.append("removed_leading_think_block")

    cut_candidates = []
    transcript_match = ROLE_MARKER_RE.search(cleaned)
    if transcript_match:
        cut_candidates.append((transcript_match.start(), "truncated_role_transcript"))

    for marker, action in (
        ("### Instruction:", "truncated_instruction_leak"),
        ("<|assistant|>", "truncated_special_role_token"),
        ("<|user|>", "truncated_special_role_token"),
        ("<|system|>", "truncated_special_role_token"),
    ):
        marker_idx = cleaned.lower().find(marker.lower())
        if marker_idx != -1:
            cut_candidates.append((marker_idx, action))

    if cut_candidates:
        cut_idx, action = min(cut_candidates, key=lambda item: item[0])
        cleaned = cleaned[:cut_idx]
        cleanup_actions.append(action)

    return cleaned.strip(), cleanup_actions


def find_answer_flags(answer: str, settings: dict):
    if not isinstance(answer, str):
        return ["non_string_answer"]

    flags = []
    stripped = answer.strip()
    min_chars = int(settings.get("sanity_min_chars", DEFAULT_SANITY_MIN_CHARS))

    if not stripped:
        flags.append("empty")
    elif len(stripped) < min_chars:
        flags.append("too_short")

    lowered = stripped.lower()
    if "<think>" in lowered or "</think>" in lowered:
        flags.append("contains_think_tag")
    if ROLE_MARKER_RE.search(stripped):
        flags.append("contains_role_marker")

    for marker, flag in (
        ("### Instruction:", "contains_instruction_leak"),
        ("<|assistant|>", "contains_special_role_token"),
        ("<|user|>", "contains_special_role_token"),
        ("<|system|>", "contains_special_role_token"),
    ):
        if marker.lower() in lowered:
            flags.append(flag)

    for marker in settings.get("sanity_disallowed_substrings", []):
        if marker.lower() in lowered:
            flags.append(f"contains_marker:{marker}")

    return list(dict.fromkeys(flags))


def dump_rejected_answer(answer_file: str, question: dict, raw_answer, cleaned_answer, flags, cleanup_actions, attempt):
    reject_file = f"{answer_file}.rejects"
    payload = {
        "uid": question["uid"],
        "model": model,
        "attempt": attempt,
        "flags": flags,
        "cleanup_actions": cleanup_actions,
        "raw_answer": raw_answer,
        "cleaned_answer": cleaned_answer,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(reject_file), exist_ok=True)
    with DEBUG_DUMP_LOCK:
        with open(reject_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def dump_debug_attempt(answer_file: str, question: dict, messages: list, settings: dict, attempt: int, max_attempts: int, raw_answer, cleaned_answer, flags, cleanup_actions, accepted: bool):
    if not settings.get("debug_dump_attempts"):
        return

    debug_dir = settings.get("debug_dump_dir") or os.path.join(os.path.dirname(answer_file), "debug_attempts")
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = os.path.join(debug_dir, f"{model}.attempts.jsonl")
    payload = {
        "uid": question["uid"],
        "category": question.get("category"),
        "model": model,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "accepted": accepted,
        "flags": flags,
        "cleanup_actions": cleanup_actions,
        "messages": messages,
        "raw_answer": raw_answer,
        "cleaned_answer": cleaned_answer,
        "tstamp": time.time(),
    }

    with DEBUG_DUMP_LOCK:
        with open(debug_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")


def get_answer(
    question: dict, answer_file: str, settings: dict
):
    # build messages
    messages = []
    if "sys_prompt" in settings:
        messages.append({"role": "system", "content": settings["sys_prompt"]})
        
    messages.append({"role": "user", "content": question["prompt"]})

    # retrieve the api completion function from register
    api_completion_func = registered_api_completion[settings["api_type"]]
    
    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }

    sanitize_output = bool(settings.get("sanitize_output", False))
    sanity_check = bool(settings.get("sanity_check", False))
    max_attempts = 1 + int(settings.get("sanity_max_retries", 0)) if sanity_check else 1

    output = None
    quality_flags = []
    cleanup_actions = []
    last_rejection = None

    for attempt in range(1, max_attempts + 1):
        candidate = api_completion_func(**kwargs)
        if candidate is API_ERROR_OUTPUT:
            dump_debug_attempt(
                answer_file,
                question,
                messages,
                settings,
                attempt,
                max_attempts,
                raw_answer=None,
                cleaned_answer=None,
                flags=["api_error"],
                cleanup_actions=[],
                accepted=False,
            )
            last_rejection = {
                "raw_answer": None,
                "cleaned_answer": None,
                "flags": ["api_error"],
                "cleanup_actions": [],
                "attempt": attempt,
            }
            continue

        raw_answer = candidate.get("answer", "")
        cleaned_answer = raw_answer
        candidate_cleanup_actions = []
        if sanitize_output:
            cleaned_answer, candidate_cleanup_actions = sanitize_answer_text(raw_answer)
            candidate = candidate | {"answer": cleaned_answer}

        candidate_flags = find_answer_flags(cleaned_answer, settings) if (sanitize_output or sanity_check) else []
        dump_debug_attempt(
            answer_file,
            question,
            messages,
            settings,
            attempt,
            max_attempts,
            raw_answer=raw_answer,
            cleaned_answer=cleaned_answer,
            flags=candidate_flags,
            cleanup_actions=candidate_cleanup_actions,
            accepted=not (sanity_check and candidate_flags),
        )
        if sanity_check and candidate_flags:
            print(
                f"[sanity] rejected answer for model={model} uid={question['uid']} "
                f"attempt={attempt}/{max_attempts} flags={candidate_flags}"
            )
            last_rejection = {
                "raw_answer": raw_answer,
                "cleaned_answer": cleaned_answer,
                "flags": candidate_flags,
                "cleanup_actions": candidate_cleanup_actions,
                "attempt": attempt,
            }
            continue

        output = candidate
        quality_flags = candidate_flags
        cleanup_actions = candidate_cleanup_actions
        break

    if output is None:
        if last_rejection is not None:
            dump_rejected_answer(
                answer_file,
                question,
                last_rejection["raw_answer"],
                last_rejection["cleaned_answer"],
                last_rejection["flags"],
                last_rejection["cleanup_actions"],
                last_rejection["attempt"],
            )
        return

    messages.append({"role": "assistant", "content": output})

    # Dump answers
    ans = {
        "uid": question["uid"],
        "ans_id": shortuuid.uuid(),
        "model": model,
        "messages": messages,
        "tstamp": time.time(),
    }
    
    encoding = tiktoken.encoding_for_model("gpt-4o")
    metadata = {
        "token_len": len(encoding.encode(output['answer'], disallowed_special=()))
    }
    if sanitize_output or sanity_check:
        metadata["answer_quality"] = {
            "sanity_check": sanity_check,
            "sanitize_output": sanitize_output,
            "passed": len(quality_flags) == 0,
            "flags": quality_flags,
            "cleanup_actions": cleanup_actions,
        }
    ans["metadata"] = metadata | count_markdown_elements(
        remove_pattern(
            output['answer'], 
            re.compile("```([^`]*)```")
        ),
        suffix="",
    )

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, default="config/gen_answer_config.yaml"
    )
    parser.add_argument(
        "--endpoint-file", type=str, default="config/api_config.yaml"
    )
    args = parser.parse_args()

    config = make_config(args.config_file)
    endpoints = make_config(args.endpoint_file)

    answer_dir = config.get("answer_dir", os.path.join("data", config["bench_name"], "model_answer"))
    existing_answer = load_model_answers(answer_dir)
    
    print(config)

    for model in config["model_list"]:
        assert model in endpoints
        endpoint_settings = endpoints[model]

        question_file = os.path.join("data", config["bench_name"], "question.jsonl")
        questions = filter_questions(load_questions(question_file), config)
        print(f"Loaded {len(questions)} questions for model={model}")

        answer_file = os.path.join(answer_dir, f"{model}.jsonl")
        print(f"Output to {answer_file}")

        if "parallel" in endpoint_settings:
            parallel = endpoint_settings["parallel"]
        else:
            parallel = 1
            
        if 'local_engine' in endpoint_settings and endpoint_settings['local_engine']:
            local_completion_func = registered_engine_completion[endpoint_settings['api_type']]
            
            kwargs = endpoint_settings | {
                "answer_file": answer_file,
                "batch_context": questions,
            }
            local_completion_func(**kwargs)
            
            reorg_answer_file(answer_file)
            
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = []
                count = 0
                for index, question in enumerate(questions):
                    if model in existing_answer and question["uid"] in existing_answer[model]:
                        count += 1
                        continue
                    future = executor.submit(
                        get_answer,
                        question,
                        answer_file,
                        endpoint_settings,
                    )
                    futures.append(future)
                if count > 0:
                    print(f"{count} number of existing answers")
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    future.result()

            reorg_answer_file(answer_file)
            
