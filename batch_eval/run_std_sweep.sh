#!/bin/bash
# Does raising the flow-matching start-noise std buy success under random
# selection, once there are 40 candidates to draw from?
#
#   7 (task, split) x {std 1.0, 1.25, 1.5, 2.0} x 50 ep, all at 8 rephrasings x
#   5 samples with uniform-random selection.
#
#   Seen   (demo_clean):      handover_block hanging_mug move_pillbottle_pad
#                             blocks_ranking_size (1.5x step limit -> 1800)
#   Unseen (demo_randomized): click_bell handover_block open_microwave
#
# Runs the split denoising loop (`--merge_mode trace`, no merging) with candidate
# logging on, like the tts sweep: the per-step x0_hat is what shows at which
# denoising step a wider start noise makes the candidates separate. std 1.0 is
# re-run rather than reused from the tts sweep so every arm is on one footing.
#
# Under random selection the mean over candidates is what is being measured, so
# a rise in success rate with std means the wider start distribution contains
# more successful chunks on average -- not merely more spread. A fall means the
# extra spread is mostly off-distribution garbage.
#
#   setsid nohup bash run_std_sweep.sh > /dev/null 2>&1 &

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib/sweep_common.sh"

REPH="${REPH:-8}"
SAMP="${SAMP:-5}"
MERGE_MODE="${MERGE_MODE:-trace}"          # "" for the stock while_loop path
LOG_CANDIDATES="${LOG_CANDIDATES:-1}"
read -r -a STDS <<< "${STDS_OVERRIDE:-1.0 1.25 1.5 2.0}"

# (task, split, step-limit scale), longest first so the workers drain together.
# Durations are from the r8s5 arm of tts_20260911_171141.
PAIRS=(
    "blocks_ranking_size demo_clean 1.5"   # 14173 s
    "open_microwave demo_randomized 1.0"   # 13737 s
    "handover_block demo_randomized 1.0"   # 10413 s
    "hanging_mug demo_clean 1.0"           #  9760 s
    "handover_block demo_clean 1.0"        #  7994 s
    "move_pillbottle_pad demo_clean 1.0"   #  4284 s
    "click_bell demo_randomized 1.0"       #  2638 s
)

sweep_init std
sweep_log "shape=${REPH}x${SAMP} stds=${STDS[*]} merge_mode=${MERGE_MODE:-stock}"

for pair in "${PAIRS[@]}"; do
    read -r task cfg scale <<< "${pair}"
    for std in "${STDS[@]}"; do
        label="${task}__${cfg}__r${REPH}s${SAMP}__std${std}"
        args=("--task_config=${cfg}" "--test_num=${EPISODES}"
              "--rephrase_num=${REPH}" "--policy_batch_inference_size=${SAMP}"
              "--noise_std=${std}")
        [[ "${scale}" != "1.0" ]] && { label="${label}__lim${scale}x"; args+=("--step_limit_scale=${scale}"); }
        [[ -n "${MERGE_MODE}" ]] && args+=("--merge_mode=${MERGE_MODE}")
        sweep_enqueue "${label}" "${task}" "${args[@]}"
    done
done

sweep_run
