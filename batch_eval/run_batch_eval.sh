#!/bin/bash
# RoboTwin 2.0 x Pi0.5 batch evaluation.
#
#   5 tasks x {Easy = demo_clean, Hard = demo_randomized} x 50 episodes
#   dispatched over 4 GPU workers via a flock-protected queue.
#
# Each job owns one policy server (JAX) and one simulator (SAPIEN) pinned to the
# same GPU: ~7.5 GB + ~5 GB of the 24 GB card. Detach with:
#   setsid nohup bash run_batch_eval.sh > /dev/null 2>&1 &

set -uo pipefail

ROBOTWIN_ROOT=/home2/junyoung/RoboTwin
POLICY_DIR="${ROBOTWIN_ROOT}/XPolicyLab/policy/Pi_05"
CKPT=RoboTwin-lerobot_v30-aloha_agilex-joint-0
ENV_CFG=aloha_agilex
ACTION=joint
SEED=0
EPISODES="${EPISODES:-50}"
# Test-time scaling knobs: candidates per step are REPHRASE_NUM * SAMPLES, produced
# by one batched forward pass. 1 and 1 is the unscaled baseline.
REPHRASE_NUM="${REPHRASE_NUM:-1}"
SAMPLES="${SAMPLES:-1}"
# The policy server and the simulator share a card. JAX preallocates this fraction;
# 0.3 is only enough for one action sample, and a server that runs out answers
# get_action with RESOURCE_EXHAUSTED, which the rollout counts as a failed episode.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}"
# One entry per concurrent job. "N" runs the policy server and the simulator both on
# card N; "P:E" splits them, which is what test-time scaling needs once the policy no
# longer fits alongside a simulator. "0:1 2:3" therefore runs two jobs, not four.
read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
STAGGER="${STAGGER:-15}"

RUN_DIR="${ROBOTWIN_ROOT}/batch_eval/runs/reph${REPHRASE_NUM}-samp${SAMPLES}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/args"
ln -sfn "${RUN_DIR}" "${ROBOTWIN_ROOT}/batch_eval/runs/latest"

# Longest-processing-time first: blocks_ranking_* carry a 1200-step limit, the
# rest 400, and demo_randomized episodes fail more often (= run to the limit).
# Front-loading them keeps the four workers finishing at roughly the same time.
cat > "${RUN_DIR}/queue.txt" <<'JOBS'
blocks_ranking_size	demo_randomized
blocks_ranking_rgb	demo_randomized
blocks_ranking_size	demo_clean
blocks_ranking_rgb	demo_clean
beat_block_hammer	demo_randomized
click_alarmclock	demo_randomized
click_bell	demo_randomized
beat_block_hammer	demo_clean
click_alarmclock	demo_clean
click_bell	demo_clean
JOBS
cp "${RUN_DIR}/queue.txt" "${RUN_DIR}/queue.all.txt"
printf 'job\tgpu\tstart\tend\tsecs\trc\tresult\n' > "${RUN_DIR}/status.tsv"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "${RUN_DIR}/progress.log"; }

pop_job() {
    exec 9>>"${RUN_DIR}/queue.lock"
    flock 9
    local line
    line=$(head -n 1 "${RUN_DIR}/queue.txt")
    [[ -n "${line}" ]] && sed -i '1d' "${RUN_DIR}/queue.txt"
    flock -u 9
    exec 9>&-
    printf '%s\n' "${line}"
}

worker() {
    local slot=$1 job task cfg label argsf t0 t1 rc res
    local policy_gpu="${slot%%:*}"
    local env_gpu="${slot##*:}"
    local gpu="${slot//:/+}"          # label only; keeps log lines and status.tsv readable
    while :; do
        job=$(pop_job)
        [[ -z "${job}" ]] && break
        task=$(cut -f1 <<< "${job}")
        cfg=$(cut -f2 <<< "${job}")
        label="${task}__${cfg}"
        argsf="${RUN_DIR}/args/${label}.txt"
        # eval.sh exposes none of these; eval_policy.sh appends them from this file.
        printf -- '--task_config=%s\n--test_num=%s\n--rephrase_num=%s\n--policy_batch_inference_size=%s\n' \
            "${cfg}" "${EPISODES}" "${REPHRASE_NUM}" "${SAMPLES}" > "${argsf}"

        t0=$(date +%s)
        log "GPU${gpu} START ${label}"
        (
            cd "${POLICY_DIR}" || exit 1
            export ROBOTWIN_EVAL_ARGS_FILE="${argsf}"
            export CUDA_HOME=/usr/local/cuda
            export PYTHONUNBUFFERED=1
            bash eval.sh RoboTwin "${task}" "${CKPT}" "${ENV_CFG}" "${ACTION}" \
                "${SEED}" "${policy_gpu}" "${env_gpu}" uv RoboTwin
        ) > "${RUN_DIR}/logs/${label}.log" 2>&1
        rc=$?
        t1=$(date +%s)
        res=$(grep -ohE 'Final batch success rate: .*' "${RUN_DIR}/logs/${label}.log" | tail -1)

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${label}" "${gpu}" "$(date -d "@${t0}" '+%F %T')" "$(date -d "@${t1}" '+%F %T')" \
            "$((t1 - t0))" "${rc}" "${res:-NONE}" >> "${RUN_DIR}/status.tsv"
        log "GPU${gpu} END   ${label} rc=${rc} $((t1 - t0))s ${res:-no-result}"
    done
    log "GPU${gpu} worker drained"
}

log "run dir ${RUN_DIR}"
log "gpus=${GPUS[*]} episodes=${EPISODES} jobs=$(wc -l < "${RUN_DIR}/queue.all.txt")"
log "rephrase=${REPHRASE_NUM} samples=${SAMPLES} -> $((REPHRASE_NUM * SAMPLES)) candidates/step"
log "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
for slot in "${GPUS[@]}"; do
    worker "${slot}" &
    sleep "${STAGGER}"   # avoid free-port races and 4 simultaneous JAX inits
done
wait
log "ALL DONE"
