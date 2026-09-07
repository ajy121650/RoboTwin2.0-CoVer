#!/bin/bash
# RoboTwin x Pi0.5 test-time scaling sweeps.
#
#   A  5 tasks x {clean,randomized} x rephrase{4,8,12,16,20} x samples 5   50 jobs
#   B  5 tasks x {clean,randomized} x rephrase 8 x samples{1,7,10}         30 jobs
#   C  4 new tasks x {clean,randomized} x {1x1, 4x5, 8x5, 12x5}            32 jobs
#
# One queue for all of it, drained by the GPU workers, so a card never idles at a
# sweep boundary. Sweeps are enqueued A -> C -> B, longest tasks first within each:
# the rephrase sweep answers the open question, the new tasks come next, and the
# sample-size sweep is last because it varies the weaker of the two axes.
#
#   setsid nohup bash run_sweep.sh > /dev/null 2>&1 &

set -uo pipefail

ROBOTWIN_ROOT=/home2/junyoung/RoboTwin
POLICY_DIR="${ROBOTWIN_ROOT}/XPolicyLab/policy/Pi_05"
CKPT=RoboTwin-lerobot_v30-aloha_agilex-joint-0
ENV_CFG=aloha_agilex
ACTION=joint
SEED=0
EPISODES="${EPISODES:-50}"
read -r -a GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
STAGGER="${STAGGER:-15}"
# Verified on this host: the policy peaks at 8.19 GB for the heaviest config (20x5)
# and the simulator, whose texture cache keeps growing with episode count, needs the
# rest. 0.36 ran 20x5 end to end and leaves the simulator what the 500-episode run had.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.36}"

TASKS_OLD=(blocks_ranking_size blocks_ranking_rgb beat_block_hammer click_alarmclock click_bell)
TASKS_NEW=(hanging_mug handover_block move_can_pot move_pillbottle_pad)
CONFIGS=(demo_randomized demo_clean)

RUN_DIR="${ROBOTWIN_ROOT}/batch_eval/runs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/args"
ln -sfn "${RUN_DIR}" "${ROBOTWIN_ROOT}/batch_eval/runs/latest"

emit() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "${RUN_DIR}/queue.txt"; }

: > "${RUN_DIR}/queue.txt"
for r in 20 16 12 8 4; do                       # longest first
    for t in "${TASKS_OLD[@]}"; do for c in "${CONFIGS[@]}"; do emit A "$t" "$c" "$r" 5; done; done
done
for rs in "12 5" "8 5" "4 5" "1 1"; do
    set -- $rs
    for t in "${TASKS_NEW[@]}"; do for c in "${CONFIGS[@]}"; do emit C "$t" "$c" "$1" "$2"; done; done
done
for s in 10 7 1; do
    for t in "${TASKS_OLD[@]}"; do for c in "${CONFIGS[@]}"; do emit B "$t" "$c" 8 "$s"; done; done
done
cp "${RUN_DIR}/queue.txt" "${RUN_DIR}/queue.all.txt"
printf 'sweep\tjob\tgpu\tstart\tend\tsecs\trc\tresult\n' > "${RUN_DIR}/status.tsv"

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
    local slot=$1 job sweep task cfg reph samp label argsf t0 t1 rc res
    local policy_gpu="${slot%%:*}" env_gpu="${slot##*:}" gpu="${slot//:/+}"
    while :; do
        job=$(pop_job); [[ -z "${job}" ]] && break
        IFS=$'\t' read -r sweep task cfg reph samp <<< "${job}"
        label="${sweep}_${task}__${cfg}__r${reph}s${samp}"
        argsf="${RUN_DIR}/args/${label}.txt"
        printf -- '--task_config=%s\n--test_num=%s\n--rephrase_num=%s\n--policy_batch_inference_size=%s\n' \
            "${cfg}" "${EPISODES}" "${reph}" "${samp}" > "${argsf}"

        t0=$(date +%s)
        log "GPU${gpu} START ${label}  (남은 큐 $(wc -l < "${RUN_DIR}/queue.txt"))"
        (
            cd "${POLICY_DIR}" || exit 1
            export ROBOTWIN_EVAL_ARGS_FILE="${argsf}"
            export CUDA_HOME=/usr/local/cuda
            bash eval.sh RoboTwin "${task}" "${CKPT}" "${ENV_CFG}" "${ACTION}" \
                "${SEED}" "${policy_gpu}" "${env_gpu}" uv RoboTwin
        ) > "${RUN_DIR}/logs/${label}.log" 2>&1
        rc=$?
        t1=$(date +%s)
        res=$(grep -ohE 'Final batch success rate: .*' "${RUN_DIR}/logs/${label}.log" | tail -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${sweep}" "${label}" "${gpu}" \
            "$(date -d "@${t0}" '+%F %T')" "$(date -d "@${t1}" '+%F %T')" \
            "$((t1 - t0))" "${rc}" "${res:-NONE}" >> "${RUN_DIR}/status.tsv"
        log "GPU${gpu} END   ${label} rc=${rc} $((t1 - t0))s ${res:-no-result}"
    done
    log "GPU${gpu} worker drained"
}

log "run dir ${RUN_DIR}"
log "gpus=${GPUS[*]} episodes=${EPISODES} jobs=$(wc -l < "${RUN_DIR}/queue.all.txt")"
log "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
for slot in "${GPUS[@]}"; do worker "${slot}" & sleep "${STAGGER}"; done
wait
log "ALL DONE"
