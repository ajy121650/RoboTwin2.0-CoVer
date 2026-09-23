# RoboTwin verifier (CoVer-style, joint space)

Contrastive verifier for test-time selection among Pi0.5 action-chunk candidates on
RoboTwin 2.0. Frozen SigLIP2 image/text towers + trainable heads; symmetric InfoNCE
between (head image, instruction) and the expert's next 50 joint actions.

Design decisions (all fixed, see the discussion log): Level-0 InfoNCE (in-batch
negatives only) · `demo_clean` 2,500 episodes / 50 tasks (= the Pi0.5 leaderboard
training set) · stride 5 · seen instructions for training, unseen held out ·
joint 14-D, chunk-start-relative deltas, symmetric quantile scaling · window 50, no
action history · head camera only · batch never holds two windows of one episode.

## Layout

| file | role |
|---|---|
| `build_dataset.py` | `demo_clean/<task>/aloha_agilex/{data,instruction}` → sample tables (`actions.npy`, `samples.jsonl`, `instructions.json`, `unseen.json`, `norm_stats.json`, `images/`) |
| `inspect_dataset.py` | statistics + contact sheet for a built set |
| `model.py` | `Verifier` (heads ported from CoVer), `build()`, `load_cover_warm_start()` |
| `data.py` | `VerifierDataset`, `EpisodeUniqueBatchSampler`, collate |
| `train.py` | DDP training (torchrun), val every N steps, `best.pt`/`last.pt`/`epochNNN.pt` |
| `eval_offline.py` | holdout retrieval: global / same-task / same-episode pools, seen + unseen |
| `configs/default.yaml` | the baseline run |

Every departure from CoVer is a switch, and `configs/cover.yaml` sets all of them back
(no negative gathering, no action position embedding, `token_scale: none`,
`text_mask: false`, decay on every parameter, lr 1e-6) for a faithful in-domain
re-training baseline. The two head-side switches in `configs/default.yaml`:
`text_mask: true` keeps pad tokens out of text pooling, and `token_scale: sqrt_dim`
rescales the L2-normalised SigLIP2 tokens to unit per-dim scale. Without the latter
the pooled context is input-independent at init (pairwise cosine 1.000 across
different images and texts, because the learned query and pos_emb dwarf the 0.03-scale
tokens); with it the cosine starts at ~0.9. Set both to CoVer's values (`false`,
`none`) when warm-starting from `cover_verifier_bridge.pt`.

## Setup (cloud)

```bash
pip install -r verifier/requirements.txt          # torch 2.4+; pin transformers<5 if torch<2.5
export HF_HOME=/big/disk/hf_cache                  # SigLIP2-L weights ~3.3 GB; required
```

Data loading: images are stored as the original 320x240 JPEGs and decoded + resized
to 384x384 in DataLoader workers (CPU) every step, like any CLIP training run; the
frozen SigLIP2 forward and everything after it run on the GPU. Budget ~8 worker
processes per GPU (`train.num_workers`). Resume (`--resume`) restarts at the
beginning of the epoch recorded in `last.pt`.

Copy only `verifier_data/demo_clean_s5_w50/` (1.9 GB); the hdf5 are not needed for
training. To rebuild from hdf5 elsewhere:

```bash
python verifier/build_dataset.py --data-root <...>/data/demo_clean --out <...>/verifier_data/demo_clean_s5_w50
python verifier/inspect_dataset.py <...>/verifier_data/demo_clean_s5_w50 --out contact.png
```

## Train

```bash
# 8 GPUs
torchrun --nproc_per_node 8 verifier/train.py --config verifier/configs/default.yaml \
    --data <...>/verifier_data/demo_clean_s5_w50 --out <...>/verifier_ckpt/vitl_scratch
# warm start of text/vision heads from CoVer (action encoder always fresh)
torchrun --nproc_per_node 8 verifier/train.py ... --set warm_start=/path/cover_verifier_bridge.pt --out .../vitl_warm
# smoke
python verifier/train.py --set model.backbone=hf-hub:timm/ViT-B-16-SigLIP2-256 --set train.max_steps=3 --set train.batch_size=8 --out /tmp/smoke
```

Per-GPU batch 64 with 63 in-batch negatives per rank, as in CoVer
(`train.gather_negatives: true` all-gathers embeddings instead, 64 × GPUs − 1 negatives).
One epoch over 764k train pairs (cap 40k/task) ≈ 1,490 steps per rank at 8 GPUs. Logs: `out/log.jsonl`;
`--resume` continues from `last.pt`; `--set train.wandb=true` for wandb.

## Evaluate

```bash
python verifier/eval_offline.py <out>/best.pt --out <out>/eval_holdout.json
```

Reports top-1/top-5 of the true window among (a) all holdout windows, (b) windows of
the same task, (c) windows of the same episode — (c) is the closest proxy to ranking
candidates from one state. Chance levels are printed next to each.

## Checkpoint contents

`head_state_dict` (heads only, ~26M params), `model_config`, `config`, `norm_stats`
(needed at inference to map policy outputs into the training space), `step`, `epoch`.
