#!/bin/bash
# Does merging candidates at denoising step 3 cost success rate?
#
#   4 arms x 7 (task, split) x 50 ep, all at 8 rephrasings x 5 samples
#
#   mergeD3_masking_fm5      L2 clustering,     tau 1.27     (5% false-merge budget)
#   mergeD3_masking_fm1      L2 clustering,     tau 0.427    (1%)
#   mergeD3cos_masking_fm5   cosine clustering, tau 0.00519  (5%)
#   mergeD3cos_masking_fm1   cosine clustering, tau 0.000578 (1%)
#
# The thresholds are TAU_UNIFIED in openpi/models/merge.py: one set for all
# seven pairs, calibrated on their pooled trace dumps with tau_final at CoVer's
# percentile (p78.4) of the final pairwise distances. The metric decides both
# the clustering and which member is the medoid.
#
# No control arm runs here. The unmerged reference is the `trace` r8s5 jobs of
# tts_20260911_171141: the same split denoising loop, seed and episodes. It is
# exact for the seen pairs; the unseen pairs there drew from 32 unseen
# phrasings, before the instruction lists went back to 40.
#
# Masking keeps all 40 rows and overwrites each with its representative, so it
# saves no compute -- it selects exactly what shrinking would, while staying
# bit-comparable with the unmerged loop up to step 3.
#
# Longest pairs first, and the four arms of a pair sit together, so the workers
# finish at about the same time and an early stop leaves complete sets.
#
#   setsid nohup bash run_merge_sweep.sh > /dev/null 2>&1 &
#   RETRY_FROM=<run dir> bash run_merge_sweep.sh     # only what failed there

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib/sweep_common.sh"

REPH="${REPH:-8}"
SAMP="${SAMP:-5}"
read -r -a ARMS <<< "${ARMS_OVERRIDE:-mergeD3_masking_fm5 mergeD3_masking_fm1 mergeD3cos_masking_fm5 mergeD3cos_masking_fm1}"

# (task, split, step-limit scale), longest first; durations as in run_std_sweep.
PAIRS=(
    "blocks_ranking_size demo_clean 1.5"
    "open_microwave demo_randomized 1.0"
    "handover_block demo_randomized 1.0"
    "hanging_mug demo_clean 1.0"
    "handover_block demo_clean 1.0"
    "move_pillbottle_pad demo_clean 1.0"
    "click_bell demo_randomized 1.0"
)

sweep_init merge
sweep_log "arms=${ARMS[*]}  candidates/call = ${REPH} x ${SAMP}"

for pair in "${PAIRS[@]}"; do
    read -r task cfg scale <<< "${pair}"
    for arm in "${ARMS[@]}"; do
        label="${arm}__${task}__${cfg}"
        args=("--task_config=${cfg}" "--test_num=${EPISODES}"
              "--rephrase_num=${REPH}" "--policy_batch_inference_size=${SAMP}"
              "--merge_mode=${arm}")
        [[ "${scale}" != "1.0" ]] && { label="${label}__lim${scale}x"; args+=("--step_limit_scale=${scale}"); }
        sweep_enqueue "${label}" "${task}" "${args[@]}"
    done
done

sweep_run
