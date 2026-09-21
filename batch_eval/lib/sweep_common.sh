# Shared plumbing for the batch_eval sweeps.
#
# Every sweep here is the same machine: a queue of evaluation jobs, one worker
# per GPU draining it under a lock, one status row per job, and the same guards
# in front of it. Only the jobs differ, so that is all an experiment script
# writes:
#
#   source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib/sweep_common.sh"
#   sweep_init tts                                    # run dir, guards, logging
#   sweep_enqueue "<label>" "<task>" --task_config=demo_clean --test_num=50
#   sweep_run                                         # workers, wait, ALL DONE
#
# Optional knobs, all read from the environment: EPISODES, GPUS_OVERRIDE,
# STAGGER, NEED_GB, LOG_CANDIDATES, RUN_TAG, RETRY_FROM, TMPDIR_OVERRIDE,
# ALLOW_BUSY_GPUS, DRY_RUN, XLA_PYTHON_CLIENT_MEM_FRACTION.

set -uo pipefail

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home2/junyoung/RoboTwin}"
POLICY_DIR="${ROBOTWIN_ROOT}/XPolicyLab/policy/Pi_05"
CKPT="${CKPT:-RoboTwin-lerobot_v30-aloha_agilex-joint-0}"
ENV_CFG="${ENV_CFG:-aloha_agilex}"
ACTION="${ACTION:-joint}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-50}"
STAGGER="${STAGGER:-15}"
read -r -a SWEEP_GPUS <<< "${GPUS_OVERRIDE:-0 1 2 3}"
# Measured peak for the policy is 7.82 GB at 16x5; the simulator's texture cache
# keeps growing with episode count and needs the rest of the card.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.36}"
# The policy server imports absl, which reads tempfile.gettempdir() at import and
# dies if nothing is writable. Root is shared here and has hit 100% mid-sweep,
# taking ten jobs with it, so keep temp files where the outputs already live.
export TMPDIR="${TMPDIR_OVERRIDE:-/work2/junyoung/tmp}"

SWEEP_RETRY_LABELS=""
SWEEP_DUMP_CONTROL=""

sweep_log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "${RUN_DIR}/progress.log"; }

sweep_init() {                                    # $1 = run-directory prefix
    local prefix=$1 cand avail_gb busy
    mkdir -p "${TMPDIR}" || { echo "ABORT: cannot create TMPDIR ${TMPDIR}" >&2; exit 1; }

    RUN_DIR="${ROBOTWIN_ROOT}/batch_eval/runs/${prefix}${RUN_TAG:-}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/args"
    ln -sfn "${RUN_DIR}" "${ROBOTWIN_ROOT}/batch_eval/runs/latest_${prefix}${RUN_TAG:-}"
    : > "${RUN_DIR}/queue.txt"
    printf 'job\tgpu\tstart\tend\tsecs\trc\tresult\tmode\n' > "${RUN_DIR}/status.tsv"
    sweep_log "run dir ${RUN_DIR}"
    sweep_log "gpus=${SWEEP_GPUS[*]} episodes=${EPISODES} tmpdir=${TMPDIR}"

    # RETRY_FROM=<run dir> queues only the jobs that failed there, under the same
    # labels, so the two runs read as one. The run has to be one of this
    # runner's: it reads the status.tsv columns written below.
    if [[ -n "${RETRY_FROM:-}" ]]; then
        SWEEP_RETRY_LABELS=$(awk -F'\t' 'NR>1 && ($6 != "0" || $7 == "NONE") {print $1}' \
                             "${RETRY_FROM}/status.tsv")
        [[ -z "${SWEEP_RETRY_LABELS}" ]] && { echo "ABORT: ${RETRY_FROM} has no failed jobs" >&2; exit 1; }
        sweep_log "retrying $(wc -l <<< "${SWEEP_RETRY_LABELS}") failed jobs from ${RETRY_FROM}"
    fi

    # Candidate logging is switched on by a control file rather than an
    # environment variable, so a sweep already running picks it up at its next
    # policy call -- and leaves no recorder behind when it stops.
    if [[ "${LOG_CANDIDATES:-0}" == "1" ]]; then
        SWEEP_DUMP_CONTROL="${ROBOTWIN_ROOT}/batch_eval/dump_candidates.json"
        cand="${ROBOTWIN_ROOT}/batch_eval/candidates/$(basename "${RUN_DIR}")"
        mkdir -p "${cand}"
        printf '{"dir": "%s"}\n' "${cand}" > "${SWEEP_DUMP_CONTROL}"
        sweep_log "candidate logging -> ${cand}"
        trap 'rm -f "${SWEEP_DUMP_CONTROL}"' EXIT
    fi

    # A sweep that fills the disk wastes every job after that point, and a card
    # someone else is using takes both runs down.
    avail_gb=$(df --output=avail -BG "${ROBOTWIN_ROOT}/batch_eval/runs" | tail -1 | tr -dc '0-9')
    if (( avail_gb < ${NEED_GB:-20} )); then
        echo "ABORT: only ${avail_gb}GB free where runs/ lives, need ${NEED_GB:-20}GB" >&2
        exit 1
    fi
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
           awk -F', ' -v want=" ${SWEEP_GPUS[*]//:/ } " '$2 > 500 && index(want, " "$1" ")' | cut -d, -f1)
    if [[ -n "${busy}" && "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
        echo "ABORT: GPU(s) already in use: $(tr '\n' ' ' <<< "${busy}")" >&2
        echo "       set ALLOW_BUSY_GPUS=1 to run anyway." >&2
        exit 1
    fi
}

sweep_enqueue() {                                 # <label> <task> <eval flag>...
    local label=$1 task=$2
    shift 2
    if [[ -n "${SWEEP_RETRY_LABELS}" ]]; then
        grep -qxF "${label}" <<< "${SWEEP_RETRY_LABELS}" || return 0
    fi
    printf '%s\t%s\t%s\n' "${label}" "${task}" "$*" >> "${RUN_DIR}/queue.txt"
}

sweep_pop() {
    exec 9>>"${RUN_DIR}/queue.lock"
    flock 9
    local line
    line=$(head -n 1 "${RUN_DIR}/queue.txt")
    [[ -n "${line}" ]] && sed -i '1d' "${RUN_DIR}/queue.txt"
    flock -u 9
    exec 9>&-
    printf '%s\n' "${line}"
}

sweep_worker() {
    local slot=$1 job label task args argsf t0 t1 rc res mode
    local policy_gpu="${slot%%:*}" env_gpu="${slot##*:}" gpu="${slot//:/+}"
    while :; do
        job=$(sweep_pop); [[ -z "${job}" ]] && break
        IFS=$'\t' read -r label task args <<< "${job}"
        argsf="${RUN_DIR}/args/${label}.txt"
        # Deliberate word splitting: the flags arrive space-separated and
        # eval.sh reads them one per line.
        printf '%s\n' ${args} > "${argsf}"

        t0=$(date +%s)
        sweep_log "GPU${gpu} START ${label}  (queue left $(wc -l < "${RUN_DIR}/queue.txt"))"
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
        res=$(grep -ohaE 'Final batch success rate: .*' "${RUN_DIR}/logs/${label}.log" | tail -1)
        # A merge mode that never reaches the policy looks exactly like a normal
        # run, so record whether the server announced one.
        mode=$(grep -ohaE "merge mode '[^']+' active \([^)]*\)" "${RUN_DIR}/logs/${label}.log" | tail -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${label}" "${gpu}" \
            "$(date -d "@${t0}" '+%F %T')" "$(date -d "@${t1}" '+%F %T')" \
            "$((t1 - t0))" "${rc}" "${res:-NONE}" "${mode:--}" >> "${RUN_DIR}/status.tsv"
        sweep_log "GPU${gpu} END   ${label} rc=${rc} $((t1 - t0))s ${res:-no-result}"
    done
    sweep_log "GPU${gpu} worker drained"
}

sweep_run() {
    local slot
    cp "${RUN_DIR}/queue.txt" "${RUN_DIR}/queue.all.txt"
    sweep_log "jobs=$(wc -l < "${RUN_DIR}/queue.all.txt")"

    # Freeze what this was run with. The policy code is uncommitted more often
    # than not, so record the hashes and the diff, not a revision alone.
    {
        printf 'host   %s\ndate   %s\n' "$(hostname)" "$(date '+%F %T')"
        printf 'rev    %s\n' "$(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD 2>/dev/null)"
        printf 'branch %s\n\n' "$(git -C "${ROBOTWIN_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        printf 'sha256:\n'
        sha256sum "${POLICY_DIR}/model.py" "${POLICY_DIR}/merge_sampler.py" \
                  "${POLICY_DIR}/candidate_dump.py" \
                  "${POLICY_DIR}/openpi/src/openpi/models/merge.py" \
                  "${POLICY_DIR}/openpi/src/openpi/models/pi0.py" \
                  "${ROBOTWIN_ROOT}/scripts/eval_policy_xpolicylab.py" 2>/dev/null
    } > "${RUN_DIR}/provenance.txt"
    git -C "${ROBOTWIN_ROOT}" diff > "${RUN_DIR}/uncommitted.diff" 2>/dev/null
    git -C "${ROBOTWIN_ROOT}/XPolicyLab" diff > "${RUN_DIR}/uncommitted.XPolicyLab.diff" 2>/dev/null

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        sweep_log "DRY_RUN: queue built, no workers started"
        cat "${RUN_DIR}/queue.all.txt"
        return 0
    fi
    for slot in "${SWEEP_GPUS[@]}"; do
        sweep_worker "${slot}" &
        sleep "${STAGGER}"
    done
    wait
    sweep_log "ALL DONE"
}
