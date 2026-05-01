#!/usr/bin/env bash
set -euo pipefail

# Full Arena-Hard matrix runner for a fixed 9-model sweep.
# It is resumable:
# - Existing answers are skipped by gen_answer.py
# - Existing judgments are skipped by gen_judgment.py
#
# Default behavior:
# - Runs both arena-hard-v0.1 and arena-hard-v2.0
# - Generates answers into the benchmark's standard model_answer directory
# - Backfills markdown/style metadata in-place
# - Judges with GPT-4.1
# - Prints leaderboard commands for Hard Prompt + Style Control
#
# Key environment variables:
#   BENCHES="arena-hard-v0.1 arena-hard-v2.0"
#   PORT=8000
#   API_KEY=token-abc123
#   TP=1
#   PARALLEL=8
#   READY_TIMEOUT=900
#   MAX_TOKENS_QWEN=2048
#   MAX_TOKENS_LLAMA=2048
#   JUDGE_MODEL=gpt-4.1
#   JUDGE_ENDPOINT_FILE=config/api_config.yaml
#   CHAT_TEMPLATE=/abs/path/to/chat_template.jinja
#   TRUST_REMOTE_CODE=1
#   SKIP_INFERENCE=0
#   SKIP_METADATA=0
#   SKIP_JUDGMENT=0
#   DRY_RUN=0

BENCHES="${BENCHES:-arena-hard-v0.1 arena-hard-v2.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-token-abc123}"
TP="${TP:-1}"
PARALLEL="${PARALLEL:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
MAX_TOKENS_QWEN="${MAX_TOKENS_QWEN:-2048}"
MAX_TOKENS_LLAMA="${MAX_TOKENS_LLAMA:-2048}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE:-config/api_config.yaml}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
SKIP_METADATA="${SKIP_METADATA:-0}"
SKIP_JUDGMENT="${SKIP_JUDGMENT:-0}"
DRY_RUN="${DRY_RUN:-0}"

STOP_SEQUENCES='["\nuser\n", "\nassistant\n", "\nsystem\n", "<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "### Instruction:"]'
GENERATED_DIR="config/generated/matrix_eval"
LLAMA3_CHAT_TEMPLATE="${LLAMA3_CHAT_TEMPLATE:-templates/llama3_chat.jinja}"
QWEN3_CHAT_TEMPLATE="${QWEN3_CHAT_TEMPLATE:-templates/qwen3_nonthinking_chat.jinja}"

models=(
  "llama3_8b_sft=W-61/llama-3-8b-base-sft-ultrachat-8xh200"
  "llama3_8b_margin_dpo=W-61/llama-3-8b-base-margin-dpo-ultrafeedback-8xh200"
  "llama3_8b_beta_dpo=W-61/llama-3-8b-base-beta-dpo-ultrafeedback-4xh200-batch-128-20260424-044124"
  "llama3_8b_simpo=jackf857/llama-3-8b-base-simpo-8xh200"
  "llama3_8b_new_dpo=W-61/llama-3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-s_star-0.4-20260425-111846"
  "qwen3_8b_sft=jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128"
  "qwen3_8b_margin_dpo=W-61/qwen3-8b-base-margin-dpo-ultrafeedback-4xh200-batch-128-20260423-040315"
  "qwen3_8b_beta_dpo=W-61/qwen3-8b-base-beta-dpo-ultrafeedback-4xh200-batch-128-20260423-040315"
  "qwen3_8b_simpo=jackf857/qwen3-8b-base-simpo-ultrafeedback-4xH200-batch-128"
)

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return 0
  fi

  "$@"
}

write_gen_config() {
  local bench="$1"
  local alias="$2"
  local output_file="$3"
  cat > "${output_file}" <<YAML
bench_name: ${bench}

model_list:
  - ${alias}
YAML
}

write_api_config() {
  local alias="$1"
  local max_tokens="$2"
  local output_file="$3"
  cat > "${output_file}" <<YAML
${alias}:
  model: ${alias}
  endpoints:
    - api_base: http://127.0.0.1:${PORT}/v1
      api_key: ${API_KEY}
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${max_tokens}
  temperature: 0.0
  stop: ${STOP_SEQUENCES}
  sanitize_output: true
  sanity_check: true
  sanity_max_retries: 2
  sanity_min_chars: 16
YAML
}

write_judge_config() {
  local bench="$1"
  local output_file="$2"

  local max_tokens="16000"
  if [[ "${bench}" == "arena-hard-v0.1" ]]; then
    max_tokens="4096"
  fi

  {
    printf 'judge_model: %s\n' "${JUDGE_MODEL}"
    printf 'bench_name: %s\n' "${bench}"
    printf 'reference: null\n'
    printf 'temperature: 0.0\n'
    printf 'max_tokens: %s\n\n' "${max_tokens}"
    printf 'regex_patterns:\n'
    printf '  - \\[\\[([AB<>=]+)\\]\\]\n'
    printf '  - \\[([AB<>=]+)\\]\n\n'
    printf 'prompt_template: "<|User Prompt|>\\n{QUESTION}\\n\\n<|The Start of Assistant A'\''s Answer|>\\n{ANSWER_A}\\n<|The End of Assistant A'\''s Answer|>\\n\\n<|The Start of Assistant B'\''s Answer|>\\n{ANSWER_B}\\n<|The End of Assistant B'\''s Answer|>"\n\n'
    printf 'model_list:\n'
    for spec in "${models[@]}"; do
      printf '  - %s\n' "${spec%%=*}"
    done
  } > "${output_file}"
}

ensure_vllm() {
  if command -v vllm >/dev/null 2>&1; then
    VLLM_CMD=(vllm)
    return 0
  fi

  if uv run --no-sync python -c "import vllm" >/dev/null 2>&1; then
    VLLM_CMD=(uv run --no-sync vllm)
    return 0
  fi

  cat >&2 <<'EOF'
vLLM is not available in PATH or the current uv environment.
This runner serves one checkpoint at a time through an OpenAI-compatible vLLM endpoint.

If you want to execute the full matrix, run this script on a Linux GPU host with vLLM installed,
or adapt the endpoint config to point at already-running remote model servers.
EOF
  exit 1
}

get_chat_template_for_alias() {
  local alias="$1"

  if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
    printf '%s\n' "${CHAT_TEMPLATE}"
    return 0
  fi

  case "${alias}" in
    llama3_8b_*)
      printf '%s\n' "${LLAMA3_CHAT_TEMPLATE}"
      ;;
    qwen3_8b_*)
      printf '%s\n' "${QWEN3_CHAT_TEMPLATE}"
      ;;
    *)
      printf '\n'
      ;;
  esac
}

wait_for_server() {
  local alias="$1"
  local log_file="$2"
  local pid="$3"
  local start_ts
  start_ts="$(date +%s)"

  until curl -sf -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vLLM exited before becoming ready for ${alias}. Recent log output:" >&2
      tail -n 100 "${log_file}" >&2 || true
      exit 1
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= READY_TIMEOUT )); then
      echo "Timed out waiting ${READY_TIMEOUT}s for ${alias} to become ready." >&2
      tail -n 100 "${log_file}" >&2 || true
      exit 1
    fi
    sleep 5
  done
}

run_one_inference() {
  local bench="$1"
  local alias="$2"
  local repo="$3"
  local gen_config="$4"
  local api_config="$5"

  local max_tokens="${MAX_TOKENS_QWEN}"
  if [[ "${alias}" == llama3_8b_* ]]; then
    max_tokens="${MAX_TOKENS_LLAMA}"
  fi
  local chat_template
  chat_template="$(get_chat_template_for_alias "${alias}")"

  write_gen_config "${bench}" "${alias}" "${gen_config}"
  write_api_config "${alias}" "${max_tokens}" "${api_config}"

  if [[ "${repo}" == *"-base-"* ]]; then
    echo "WARNING: ${repo} looks like a base checkpoint. This runner will apply ${chat_template:-no chat template}."
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would serve ${repo} as ${alias} for ${bench} with template ${chat_template:-<none>}"
    run uv run python gen_answer.py --config-file "${gen_config}" --endpoint-file "${api_config}"
    return 0
  fi

  local log_file="logs/${bench}.${alias}.vllm.log"
  mkdir -p logs

  local -a vllm_extra_args=()
  if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
    vllm_extra_args+=(--trust-remote-code)
  fi
  if [[ -n "${chat_template}" ]]; then
    vllm_extra_args+=(--chat-template "${chat_template}")
  fi

  "${VLLM_CMD[@]}" serve "${repo}" \
    --served-model-name "${alias}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TP}" \
    "${vllm_extra_args[@]}" \
    > "${log_file}" 2>&1 &
  local vllm_pid="$!"

  cleanup() {
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
  }
  trap cleanup RETURN

  wait_for_server "${alias}" "${log_file}" "${vllm_pid}"
  run uv run python gen_answer.py --config-file "${gen_config}" --endpoint-file "${api_config}"
}

backfill_style_metadata() {
  local bench="$1"
  run uv run python utils/add_markdown_info.py \
    --dir "data/${bench}/model_answer" \
    --output-dir "data/${bench}/model_answer"
}

run_judgment_for_bench() {
  local bench="$1"
  local judge_config="$2"
  write_judge_config "${bench}" "${judge_config}"
  run uv run python gen_judgment.py --setting-file "${judge_config}" --endpoint-file "${JUDGE_ENDPOINT_FILE}"
}

print_score_commands() {
  local bench="$1"

  if [[ "${bench}" == "arena-hard-v0.1" ]]; then
    cat <<EOF
Leaderboard command for ${bench}:
  uv run python show_result.py --benchmark ${bench} --judge-names ${JUDGE_MODEL} --control-features markdown length --category arena-hard-v0.1
EOF
    return 0
  fi

  cat <<EOF
Leaderboard command for ${bench} Hard Prompt + Style Control:
  uv run python show_result.py --benchmark ${bench} --judge-names ${JUDGE_MODEL} --control-features markdown length --category hard_prompt
EOF
}

main() {
  mkdir -p "${GENERATED_DIR}"

  if [[ "${SKIP_INFERENCE}" != "1" && "${DRY_RUN}" != "1" ]]; then
    ensure_vllm
  fi

  if [[ "${SKIP_JUDGMENT}" != "1" && "${DRY_RUN}" != "1" ]]; then
    if [[ "${JUDGE_ENDPOINT_FILE}" == "config/api_config.yaml" && -z "${OPENAI_API_KEY:-}" ]]; then
      cat >&2 <<'EOF'
Judgment is enabled, but OPENAI_API_KEY is not set and JUDGE_ENDPOINT_FILE is still config/api_config.yaml.
Set OPENAI_API_KEY for the default GPT-4.1 judge path, or point JUDGE_ENDPOINT_FILE at a custom endpoint config.
EOF
      exit 1
    fi
  fi

  for bench in ${BENCHES}; do
    echo "=== Benchmark: ${bench} ==="

    for spec in "${models[@]}"; do
      alias="${spec%%=*}"
      repo="${spec#*=}"

      gen_config="${GENERATED_DIR}/${bench}.${alias}.gen_answer.yaml"
      api_config="${GENERATED_DIR}/${bench}.${alias}.api.yaml"

      if [[ "${SKIP_INFERENCE}" != "1" ]]; then
        run_one_inference "${bench}" "${alias}" "${repo}" "${gen_config}" "${api_config}"
      fi
    done

    if [[ "${SKIP_METADATA}" != "1" ]]; then
      backfill_style_metadata "${bench}"
    fi

    if [[ "${SKIP_JUDGMENT}" != "1" ]]; then
      judge_config="${GENERATED_DIR}/${bench}.${JUDGE_MODEL}.judge.yaml"
      run_judgment_for_bench "${bench}" "${judge_config}"
    fi

    print_score_commands "${bench}"
    echo
  done
}

main "$@"
