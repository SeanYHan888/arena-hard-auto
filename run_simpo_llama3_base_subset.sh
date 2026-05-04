#!/usr/bin/env bash
set -euo pipefail

# Inference-only runner for the Princeton SimPO Llama-3 base subset on both
# Arena-Hard-v0.1 and Arena-Hard-v2.0. It serves one checkpoint at a time via
# vLLM, then runs gen_answer.py against the standard benchmark question files.

BENCHES="${BENCHES:-arena-hard-v0.1 arena-hard-v2.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-token-abc123}"
TP="${TP:-1}"
PARALLEL="${PARALLEL:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SANITY_RETRIES="${SANITY_RETRIES:-2}"
SANITY_MIN_CHARS="${SANITY_MIN_CHARS:-16}"
GENERATED_DIR="${GENERATED_DIR:-config/generated/simpo_llama3_base_subset}"
LLAMA3_CHAT_TEMPLATE="${LLAMA3_CHAT_TEMPLATE:-templates/llama3_chat.jinja}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

STOP_SEQUENCES='["\nuser\n", "\nassistant\n", "\nsystem\n", "<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "### Instruction:"]'

models=(
  "llama_3_base_8b_sft_cpo=princeton-nlp/Llama-3-Base-8B-SFT-CPO"
  "llama_3_base_8b_sft_rrhf=princeton-nlp/Llama-3-Base-8B-SFT-RRHF"
  "llama_3_base_8b_sft_slic_hf=princeton-nlp/Llama-3-Base-8B-SFT-SLiC-HF"
  "llama_3_base_8b_sft_ipo=princeton-nlp/Llama-3-Base-8B-SFT-IPO"
  "llama_3_base_8b_sft_dpo=princeton-nlp/Llama-3-Base-8B-SFT-DPO"
  "llama_3_base_8b_sft_kto=princeton-nlp/Llama-3-Base-8B-SFT-KTO"
  "llama_3_base_8b_sft_orpo=princeton-nlp/Llama-3-Base-8B-SFT-ORPO"
  "llama_3_base_8b_sft_rdpo=princeton-nlp/Llama-3-Base-8B-SFT-RDPO"
  "llama_3_base_8b_sft_simpo=princeton-nlp/Llama-3-Base-8B-SFT-SimPO"
  "llama_3_base_8b_sft=princeton-nlp/Llama-3-Base-8B-SFT"
)

ensure_vllm() {
  if command -v vllm >/dev/null 2>&1; then
    VLLM_CMD=(vllm)
    return 0
  fi

  if uv run --no-sync python -c "import vllm" >/dev/null 2>&1; then
    VLLM_CMD=(uv run --no-sync vllm)
    return 0
  fi

  echo "vLLM is not available in PATH or the current uv environment." >&2
  exit 1
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
  local output_file="$2"

  cat > "${output_file}" <<YAML
${alias}:
  model: ${alias}
  endpoints:
    - api_base: http://127.0.0.1:${PORT}/v1
      api_key: ${API_KEY}
  api_type: openai
  parallel: ${PARALLEL}
  max_tokens: ${MAX_TOKENS}
  temperature: 0.0
  stop: ${STOP_SEQUENCES}
  sanitize_output: true
  sanity_check: true
  sanity_max_retries: ${SANITY_RETRIES}
  sanity_min_chars: ${SANITY_MIN_CHARS}
YAML
}

wait_for_server() {
  local alias="$1"
  local log_file="$2"
  local pid="$3"
  local start_ts
  start_ts="$(date +%s)"

  until curl -sf -H "Authorization: Bearer ${API_KEY}" \
    "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
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

run_one_model_for_bench() {
  local bench="$1"
  local alias="$2"
  local repo="$3"

  local gen_config="${GENERATED_DIR}/${bench}.${alias}.gen.yaml"
  local api_config="${GENERATED_DIR}/${bench}.${alias}.api.yaml"
  local log_file="logs/${bench}.${alias}.vllm.log"
  local -a extra_args=()

  write_gen_config "${bench}" "${alias}" "${gen_config}"
  write_api_config "${alias}" "${api_config}"

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    extra_args+=(--trust-remote-code)
  fi
  if [[ -n "${LLAMA3_CHAT_TEMPLATE}" ]]; then
    extra_args+=(--chat-template "${LLAMA3_CHAT_TEMPLATE}")
  fi

  echo "=== Benchmark ${bench}: serving ${repo} as ${alias} ==="
  "${VLLM_CMD[@]}" serve "${repo}" \
    --served-model-name "${alias}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TP}" \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &
  local vllm_pid="$!"

  cleanup() {
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
  }
  trap cleanup RETURN

  wait_for_server "${alias}" "${log_file}" "${vllm_pid}"

  uv run python gen_answer.py \
    --config-file "${gen_config}" \
    --endpoint-file "${api_config}"
}

main() {
  mkdir -p "${GENERATED_DIR}" logs
  ensure_vllm

  for bench in ${BENCHES}; do
    for spec in "${models[@]}"; do
      alias="${spec%%=*}"
      repo="${spec#*=}"
      run_one_model_for_bench "${bench}" "${alias}" "${repo}"
    done
  done
}

main "$@"
