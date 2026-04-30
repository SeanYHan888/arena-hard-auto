#!/usr/bin/env bash
set -euo pipefail

BENCH="arena-hard-v0.1"   # change to arena-hard-v2.0 if needed
PORT="8000"
API_KEY="token-abc123"
TP="1"                    # adjust for your GPU count
MAX_TOKENS="8196"
LLAMA_MAX_TOKENS="2048"
PARALLEL="8"
READY_TIMEOUT="900"

mkdir -p logs

if command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm)
elif uv run --no-sync python -c "import vllm" >/dev/null 2>&1; then
  VLLM_CMD=(uv run --no-sync vllm)
else
  echo "vllm is not installed in PATH or the current uv environment." >&2
  echo "Install it first, then rerun this script." >&2
  exit 1
fi

cat > config/api_config.local.yaml <<YAML
qwen3_8b_sft:
  model: qwen3_8b_sft
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_margin_dpo:
  model: qwen3_8b_margin_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_beta_dpo:
  model: qwen3_8b_beta_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_epsilon_dpo:
  model: qwen3_8b_epsilon_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_kto:
  model: qwen3_8b_kto
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_r_dpo:
  model: qwen3_8b_r_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_slic:
  model: qwen3_8b_slic
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_ipo:
  model: qwen3_8b_ipo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_cpo:
  model: qwen3_8b_cpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_simpo:
  model: qwen3_8b_simpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
qwen3_8b_orpo:
  model: qwen3_8b_orpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_sft:
  model: llama3_8b_sft
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_margin_dpo:
  model: llama3_8b_margin_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_beta_dpo:
  model: llama3_8b_beta_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_epsilon_dpo:
  model: llama3_8b_epsilon_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_ipo:
  model: llama3_8b_ipo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_cpo:
  model: llama3_8b_cpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_kto:
  model: llama3_8b_kto
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_orpo:
  model: llama3_8b_orpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_slic:
  model: llama3_8b_slic
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_simpo:
  model: llama3_8b_simpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_r_dpo:
  model: llama3_8b_r_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
llama3_8b_new_dpo:
  model: llama3_8b_new_dpo
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
YAML

models=(
  "qwen3_8b_sft=jackf857/qwen3-8b-base-sft-ultrachat-4xh200-batch-128"
  "qwen3_8b_margin_dpo=W-61/qwen3-8b-base-margin-dpo-ultrafeedback-4xh200-batch-128-20260423-040315"
  "qwen3_8b_beta_dpo=W-61/qwen3-8b-base-beta-dpo-ultrafeedback-4xh200-batch-128-20260423-040315"
  "qwen3_8b_epsilon_dpo=W-61/qwen3-8b-base-epsilon-dpo-ultrafeedback-4xh200-batch-128-20260422-131855"
  "qwen3_8b_kto=W-61/qwen3-8b-base-kto-ultrafeedback-4xh200-batch-128-20260426-105614"
  "qwen3_8b_r_dpo=jackf857/qwen-3-8b-base-r-dpo-ultrafeedback-4xH200-batch-128-rerun-2-runpod"
  "qwen3_8b_slic=W-61/qwen3-8b-base-slic-hf-ultrafeedback-4xh200-batch-128-20260422-131855"
  "qwen3_8b_ipo=W-61/qwen3-8b-base-ipo-ultrafeedback-4xh200-batch-128-20260422-131855"
  "qwen3_8b_cpo=W-61/qwen3-8b-base-cpo-ultrafeedback-4xh200-batch-128-20260422-131855"
  "qwen3_8b_simpo=jackf857/qwen3-8b-base-simpo-ultrafeedback-4xH200-batch-128"
  "qwen3_8b_orpo=jackf857/qwen3-8b-base-orpo-ultrafeedback-4xh200-batch-128"
  "llama3_8b_sft=W-61/llama-3-8b-base-sft-ultrachat-8xh200"
  "llama3_8b_margin_dpo=W-61/llama-3-8b-base-margin-dpo-ultrafeedback-8xh200"
  "llama3_8b_beta_dpo=W-61/llama-3-8b-base-beta-dpo-ultrafeedback-4xh200-batch-128-20260424-044124"
  "llama3_8b_epsilon_dpo=W-61/llama-3-8b-base-epsilon-dpo-ultrafeedback-8xh200"
  "llama3_8b_ipo=jackf857/llama-3-8b-base-ipo-ultrafeedback-4xh200-batch-128-rerun"
  "llama3_8b_cpo=jackf857/llama-3-8b-base-cpo-ultrafeedback-4xH200-batch-128-rerun"
  "llama3_8b_kto=jackf857/llama-3-8b-base-kto-ultrafeedback-4xh200-batch-128-20260427-194056"
  "llama3_8b_orpo=jackf857/llama-3-8b-base-orpo-ultrafeedback-4xh200-rerun"
  "llama3_8b_slic=jackf857/llama-3-8b-base-slic-hf-ultrafeedback-4xh200-batch-128-20260428-054623"
  "llama3_8b_simpo=jackf857/llama-3-8b-base-simpo-8xh200"
  "llama3_8b_r_dpo=jackf857/llama-3-8b-base-r-dpo-ultrafeedback-4xH200-batch-128-rerun-2-runpod"
  "llama3_8b_new_dpo=W-61/llama-3-8b-base-new-dpo-ultrafeedback-4xh200-batch-128-s_star-0.4-20260425-111846"
)

for spec in "${models[@]}"; do
  alias="${spec%%=*}"
  repo="${spec#*=}"

  model_max_tokens="${MAX_TOKENS}"
  if [[ "${alias}" == llama3_8b_* ]]; then
    model_max_tokens="${LLAMA_MAX_TOKENS}"
  fi

  echo "=== Serving ${repo} as ${alias} ==="

  cat > config/gen_answer_single.yaml <<YAML
bench_name: ${BENCH}
model_list:
  - ${alias}
YAML

  cat > config/api_config.local.yaml <<YAML
${alias}:
  model: ${alias}
  endpoints: [{api_base: "http://127.0.0.1:${PORT}/v1", api_key: "${API_KEY}"}]
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${model_max_tokens}
  temperature: 0.0
YAML

  "${VLLM_CMD[@]}" serve "${repo}" \
    --served-model-name "${alias}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TP}" \
    > "logs/${alias}.vllm.log" 2>&1 &
  VLLM_PID=$!

  cleanup() {
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  }
  trap cleanup EXIT

  start_ts=$(date +%s)
  until curl -sf \
    -H "Authorization: Bearer ${API_KEY}" \
    "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "vLLM exited before becoming ready for ${alias}. Recent log output:" >&2
      tail -n 200 "logs/${alias}.vllm.log" >&2 || true
      exit 1
    fi
    now_ts=$(date +%s)
    if (( now_ts - start_ts >= READY_TIMEOUT )); then
      echo "Timed out waiting ${READY_TIMEOUT}s for ${alias} to become ready." >&2
      echo "Recent vLLM log output:" >&2
      tail -n 200 "logs/${alias}.vllm.log" >&2 || true
      exit 1
    fi
    sleep 5
  done

  uv run python gen_answer.py \
    --config-file config/gen_answer_single.yaml \
    --endpoint-file config/api_config.local.yaml

  cleanup
  trap - EXIT
done
