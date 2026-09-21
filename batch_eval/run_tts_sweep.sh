#!/bin/bash
# Does test-time scaling pay, once the instruction and object wording has been
# widened?
#
#   16 tasks x {demo_clean, demo_randomized} x {1x1, 4x5, 8x5, 16x5} x 50 ep
#
# 1x1 is the unscaled baseline: one instruction, one action chunk. The other
# three hold samples at 5 and walk the rephrase axis, so a slope across them is
# the effect of having more distinct instructions to draw from.
#
# Every job runs the split denoising loop (`--merge_mode trace`), not the stock
# one. Merging can only ever run on the split path, so a merge arm measured
# against a stock baseline would differ from it by ~2e-2 for reasons unrelated
# to merging. Running the whole sweep on the path the optimisation will use
# keeps every number on one footing, and it is what makes the per-step
# extrapolations recordable at all.
#
# blocks_ranking_rgb and blocks_ranking_size get 1.5x their step limit
# (1200 -> 1800): they are the only two that regularly run out of steps rather
# than failing outright, and a truncated episode reads as a failure.
#
#   setsid nohup bash run_tts_sweep.sh > /dev/null 2>&1 &

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib/sweep_common.sh"

# TASKS_OVERRIDE lets a second sweep cover a different task set on the other
# four cards while the first one is still draining.
read -r -a TASKS <<< "${TASKS_OVERRIDE:-blocks_ranking_size blocks_ranking_rgb put_object_cabinet handover_block hanging_mug move_can_pot open_microwave pick_diverse_bottles stack_blocks_three move_stapler_pad move_pillbottle_pad place_mouse_pad click_alarmclock beat_block_hammer place_phone_stand click_bell}"
CONFIGS=(demo_randomized demo_clean)
SETTINGS=("1 1" "16 5" "8 5" "4 5")        # heaviest candidate counts first
SLOW_TASKS=(blocks_ranking_size blocks_ranking_rgb)
MERGE_MODE="${MERGE_MODE:-trace}"
LOG_CANDIDATES="${LOG_CANDIDATES:-1}"      # ~29 GB over a full sweep
NEED_GB="${NEED_GB:-40}"

sweep_init tts
sweep_log "settings=${SETTINGS[*]}  1.5x step limit on: ${SLOW_TASKS[*]}"

step_scale() {                             # $1 = task
    local s
    for s in "${SLOW_TASKS[@]}"; do [[ "$1" == "${s}" ]] && { printf '1.5'; return; }; done
    printf '1.0'
}

for setting in "${SETTINGS[@]}"; do
    read -r reph samp <<< "${setting}"
    for task in "${TASKS[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            scale=$(step_scale "${task}")
            label="${task}__${cfg}__r${reph}s${samp}"
            args=("--task_config=${cfg}" "--test_num=${EPISODES}"
                  "--rephrase_num=${reph}" "--policy_batch_inference_size=${samp}")
            [[ "${scale}" != "1.0" ]] && { label="${label}__lim${scale}x"; args+=("--step_limit_scale=${scale}"); }
            [[ -n "${MERGE_MODE}" ]] && args+=("--merge_mode=${MERGE_MODE}")
            sweep_enqueue "${label}" "${task}" "${args[@]}"
        done
    done
done

sweep_run
