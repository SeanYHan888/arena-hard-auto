#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  ./debug_inference_one_model.sh <served-model-alias> <hf-repo-or-local-path>

Example:
  ./debug_inference_one_model.sh \
    llama3_8b_new_dpo \
    W-61/llama-3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-s_star-0.4-20260425-111846

Environment overrides:
  BENCH=arena-hard-v2.0
  PORT=8000
  API_KEY=token-abc123
  TP=1
  MAX_TOKENS=2048
  TEMPERATURE=0.0
  RUN_JUDGMENT=1
  JUDGE_MODEL=gpt-4.1
  JUDGE_ENDPOINT_FILE=config/api_config.yaml
  CHAT_TEMPLATE=/abs/path/to/chat_template.jinja
  QUESTION_UIDS=uid1,uid2,uid3
  QUESTION_LIMIT=8
EOF
  exit 1
fi

ALIAS="$1"
REPO="$2"

BENCH="${BENCH:-arena-hard-v2.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-token-abc123}"
TP="${TP:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RUN_JUDGMENT="${RUN_JUDGMENT:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE:-config/api_config.yaml}"
QUESTION_UIDS="${QUESTION_UIDS:-}"
QUESTION_LIMIT="${QUESTION_LIMIT:-8}"

case "${ALIAS}" in
  llama3_8b_*)
    DEFAULT_MAX_TOKENS="2048"
    ;;
  *)
    DEFAULT_MAX_TOKENS="2048"
    ;;
esac
MAX_TOKENS="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"

ANSWER_DIR="data/${BENCH}/model_answer/debug_${ALIAS}"
DEBUG_DIR="data/${BENCH}/debug_inference"
JUDGMENT_DIR="data/${BENCH}/model_judgment/${JUDGE_MODEL}/debug_${ALIAS}"
GEN_CONFIG="config/gen_answer.debug.${ALIAS}.yaml"
API_CONFIG="config/api_config.local.debug.${ALIAS}.yaml"
JUDGE_CONFIG="config/judge.debug.${ALIAS}.yaml"
LOG_FILE="logs/${ALIAS}.debug.vllm.log"

mkdir -p logs "${ANSWER_DIR}" "${DEBUG_DIR}"

if command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm)
elif uv run --no-sync python -c "import vllm" >/dev/null 2>&1; then
  VLLM_CMD=(uv run --no-sync vllm)
else
  echo "vllm is not installed in PATH or the current uv environment." >&2
  exit 1
fi

question_yaml() {
  if [[ -n "${QUESTION_UIDS}" ]]; then
    echo "question_uids:"
    IFS=',' read -r -a uid_array <<< "${QUESTION_UIDS}"
    for uid in "${uid_array[@]}"; do
      echo "  - ${uid}"
    done
    return
  fi

  if [[ "${BENCH}" == "arena-hard-v2.0" ]]; then
    cat <<'EOF'
question_uids:
  - 2edbb5f36f5b42be
  - ec71c09662a64365
  - d5cdf24c4e614beb
  - dfc9be7c176d46bb
  - 666d2acdd7d64e17
  - d657b1fb82b141da
EOF
    return
  fi

  echo "question_limit: ${QUESTION_LIMIT}"
}

cat > "${GEN_CONFIG}" <<YAML
bench_name: ${BENCH}
answer_dir: ${ANSWER_DIR}

model_list:
  - ${ALIAS}

$(question_yaml)
YAML

cat > "${API_CONFIG}" <<YAML
${ALIAS}:
  model: ${ALIAS}
  endpoints:
    - api_base: http://127.0.0.1:${PORT}/v1
      api_key: ${API_KEY}
  api_type: openai
  parallel: 1
  max_tokens: ${MAX_TOKENS}
  temperature: ${TEMPERATURE}
  stop:
    - "\\nuser\\n"
    - "\\nassistant\\n"
    - "\\nsystem\\n"
    - "<|eot_id|>"
    - "<|im_end|>"
    - "<|end_of_text|>"
    - "### Instruction:"
  sanitize_output: true
  sanity_check: true
  sanity_max_retries: 2
  sanity_min_chars: 16
  sanity_disallowed_substrings:
    - "ERSHEY"
    - "GameObjectWithTag"
  debug_dump_attempts: true
  debug_dump_dir: ${DEBUG_DIR}
YAML

if [[ "${RUN_JUDGMENT}" == "1" ]]; then
  cat > "${JUDGE_CONFIG}" <<YAML
judge_model: ${JUDGE_MODEL}
temperature: 0.0
max_tokens: 16000

bench_name: ${BENCH}
reference: null

answer_dir: ${ANSWER_DIR}
baseline_answer_dir: data/${BENCH}/model_answer
judgment_output_dir: ${JUDGMENT_DIR}

regex_patterns:
  - \\[\\[([AB<>=]+)\\]\\]
  - \\[([AB<>=]+)\\]

prompt_template: "<|User Prompt|>\\n{QUESTION}\\n\\n<|The Start of Assistant A's Answer|>\\n{ANSWER_A}\\n<|The End of Assistant A's Answer|>\\n\\n<|The Start of Assistant B's Answer|>\\n{ANSWER_B}\\n<|The End of Assistant B's Answer|>"

model_list:
  - ${ALIAS}

$(question_yaml)
YAML
fi

VLLM_EXTRA_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  VLLM_EXTRA_ARGS+=(--trust-remote-code)
fi
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  VLLM_EXTRA_ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi

"${VLLM_CMD[@]}" serve "${REPO}" \
  --served-model-name "${ALIAS}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --tensor-parallel-size "${TP}" \
  "${VLLM_EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY_TIMEOUT="${READY_TIMEOUT:-900}"
start_ts=$(date +%s)
until curl -sf \
  -H "Authorization: Bearer ${API_KEY}" \
  "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Recent log output:" >&2
    tail -n 200 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  now_ts=$(date +%s)
  if (( now_ts - start_ts >= READY_TIMEOUT )); then
    echo "Timed out waiting ${READY_TIMEOUT}s for ${ALIAS} to become ready." >&2
    tail -n 200 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 5
done

uv run python gen_answer.py \
  --config-file "${GEN_CONFIG}" \
  --endpoint-file "${API_CONFIG}"

if [[ "${RUN_JUDGMENT}" == "1" ]]; then
  uv run python gen_judgment.py \
    --setting-file "${JUDGE_CONFIG}" \
    --endpoint-file "${JUDGE_ENDPOINT_FILE}"
fi

echo
echo "Debug run complete for ${ALIAS}"
echo "vLLM log: ${LOG_FILE}"
echo "Attempt dumps: ${DEBUG_DIR}/${ALIAS}.attempts.jsonl"
echo "Accepted answers: ${ANSWER_DIR}/${ALIAS}.jsonl"
echo "Rejected answers: ${ANSWER_DIR}/${ALIAS}.jsonl.rejects"
if [[ "${RUN_JUDGMENT}" == "1" ]]; then
  echo "Judgments: ${JUDGMENT_DIR}/${ALIAS}.jsonl"
fi
