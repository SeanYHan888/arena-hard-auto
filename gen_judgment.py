import json
import yaml
import argparse
import os
import concurrent.futures

from tqdm import tqdm

from utils.completion import (
    load_questions,
    registered_api_completion,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)

from utils.judge_utils import JUDGE_SETTINGS


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


def get_score(judgment, patterns):
    import re
    valid_scores = {
        "A>B", "A>>B", "A=B", "A<<B", "A<B",
        "B>A", "B>>A", "B=A", "B<<A", "B<A",
    }
    all_matches = []
    for pattern in patterns:
        pattern = re.compile(pattern)
        
        matches = pattern.findall(judgment.upper())
        matches = [m for m in matches if m != ""]
        all_matches.extend(matches)
        
        if len(set(matches)) > 0:
            valid_matches = [m.strip("\n") for m in matches if m.strip("\n") in valid_scores]
            if valid_matches:
                return valid_matches[-1]

    valid_matches = [m.strip("\n") for m in all_matches if m.strip("\n") in valid_scores]
    if valid_matches:
        return valid_matches[-1]
    return None


def pairwise_judgment(question, baseline, answer, reference, configs, settings):
    prompt_args = {
        "QUESTION": question['prompt'],
        "ANSWER_A": baseline["messages"][-1]["content"]['answer'],
        "ANSWER_B": answer["messages"][-1]["content"]['answer'],
    }
    
    if reference:
        prompt_args[f"REFERENCE"] = reference["messages"][-1]["content"]['answer']
        
    user_prompt = configs["prompt_template"].format(**prompt_args)
    messages = [
        {
            "role": "system", 
            "content": JUDGE_SETTINGS[question["category"]]["system_prompt"],
        },
        {
            "role": "user", 
            "content": user_prompt,
        }
    ]

    # build arguments for api completions
    kwargs = settings | {
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    kwargs['temperature'] = configs['temperature']
    kwargs['max_tokens'] = configs['max_tokens']
    
    api_completion_func = registered_api_completion[settings["api_type"]]
    output = api_completion_func(**kwargs)
    
    if output is None:
        return None

    score = get_score(output['answer'], configs["regex_patterns"])

    result = {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }
    return result


def judgment(args):
    answer = args['answer']
    baseline = args['baseline']
    
    output = {
        "uid": args['question']["uid"],
        "category": args['question']["category"],
        "judge": args['configs']['judge_model'],
        "model": answer["model"],
        "baseline": baseline["model"],
        "games": []
    }

    # round 1
    result = pairwise_judgment(
        question=args['question'],
        baseline=baseline,
        answer=answer,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)
        
    # round 2
    result = pairwise_judgment(
        question=args['question'],
        baseline=answer,
        answer=baseline,
        reference=args['reference'],
        configs=args['configs'],
        settings=args['settings'],
    )
    output["games"].append(result)

    with open(args['output_file'], "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/arena-hard-v2.0.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, reference: {configs["reference"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}')

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = configs.get("answer_dir", os.path.join("data", configs["bench_name"], "model_answer"))
    baseline_answer_dir = configs.get("baseline_answer_dir", answer_dir)

    questions = filter_questions(load_questions(question_file), configs)
    print(f"Loaded {len(questions)} questions for judgment")
    model_answers = load_model_answers(answer_dir)
    baseline_answers = model_answers if baseline_answer_dir == answer_dir else load_model_answers(baseline_answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    if configs["reference"]:
        assert not configs["reference"] in models, "ERROR: one of the models being evaluated is used as reference."
        ref_answers = [answer_dir[model] for model in configs["reference"]]
    else:
        ref_answers = None
    
    output_files = {}
    output_dir = configs.get(
        "judgment_output_dir",
        f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}",
    )
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_settings = endpoint_list[configs["judge_model"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_settings["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            if model not in model_answers:
                print(f"Warning: no answers found for model {model} in {answer_dir}")
                continue
            for question in questions:
                uid = question["uid"]

                kwargs = {}
                kwargs["question"] = question
                if uid not in model_answers[model]:
                    print(f"Warning: {model} answer to {question['uid']} cannot be found.")
                    continue

                if model in existing_judgments and uid in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][uid]
                baseline_model = JUDGE_SETTINGS[question["category"]]["baseline"]
                if baseline_model not in baseline_answers or uid not in baseline_answers[baseline_model]:
                    print(f"Warning: baseline {baseline_model} answer to {question['uid']} cannot be found.")
                    continue
                kwargs["baseline"] = baseline_answers[baseline_model][uid]
                
                if ref_answers:
                    kwargs["reference"] = [ref_answer[uid] for ref_answer in ref_answers]
                else:
                    kwargs["reference"] = None
                    
                kwargs["configs"] = configs
                kwargs["settings"] = endpoint_settings
                kwargs["output_file"] = output_files[model]
                                
                future = executor.submit(judgment, kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
