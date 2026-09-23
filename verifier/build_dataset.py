#!/usr/bin/env python3
"""Turn RoboTwin demo_clean hdf5 into a CoVer-style verifier training set.

One sample = (head-camera frame at t, one instruction, the expert's next W joint
actions expressed relative to the joint state at t). The layout mirrors the
BridgeDataset lookup tables in CoVer so the training code can stay close to the
original:

    <out>/
      images/<task>/ep<NNNNNNN>_t<TTTT>.jpg   raw JPEG bytes from the hdf5 (no re-encode)
      actions.npy          (N_windows, W, 14) float16, normalized, -5 where padded
      instructions.json    [str]  deduplicated over the whole set
      samples.jsonl        one line per (frame, instruction): ids into the tables
      unseen.json          {"<task>/<ep>": [unseen instruction strings]}   (eval only)
      norm_stats.json      q01/q99 of the delta joints, gripper mapping
      meta.json            stride, window, dims, counts, split rule

Action representation (fixed by the design discussion):
  * joint space, 14 = [L arm 6, L grip, R arm 6, R grip]  (XPolicyLab pack order)
  * arm dims are deltas against the joint STATE at t: a[t+k] - s[t]  (openpi's
    DeltaActions convention, chunk-start relative, not consecutive differences)
  * arm dims are scaled symmetrically, x / max(|q01|, |q99|), so "no motion" stays
    exactly 0 (an affine quantile map would move it); grippers are mapped 2g - 1
  * rows past the episode end are padding, value -5.0 in every dim; windows with
    fewer than --min-valid real rows are dropped as uninformative

Held-out episodes: the last `holdout_frac` of each task's episode list (by
index), so the split is deterministic and identical across runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

PAD = -5.0
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]


def pack14(g: h5py.Group) -> np.ndarray:
    """[L arm, L ee, R arm, R ee] -> (T, 14), the XPolicyLab packing order."""
    return np.concatenate(
        [g["left_arm_joint_states"][:], g["left_ee_joint_states"][:],
         g["right_arm_joint_states"][:], g["right_ee_joint_states"][:]], axis=1
    ).astype(np.float32)


def load_episode(path: pathlib.Path):
    with h5py.File(path, "r") as f:
        state = pack14(f["state"])
        action = pack14(f["action"])
        frames = f["vision/cam_head/colors"]
        jpegs = [bytes(frames[i]) for i in range(len(frames))]
        instr = json.loads(f["instructions"][()].decode()) if "instructions" in f else None
    return state, action, jpegs, instr


def windows_for_episode(state, action, stride, window, min_valid):
    """(t, delta_window (W,14) unnormalized, valid_rows) for t = 0, stride, ..."""
    T = len(action)
    out = []
    for t in range(0, T, stride):
        fut = action[t:t + window]
        n = len(fut)
        if n < min_valid:
            break
        w = np.full((window, 14), np.nan, np.float32)
        w[:n] = fut
        w[:n, ARM] -= state[t, ARM]        # chunk-start relative
        out.append((t, w, n))
    return out


def process_task(task_dir: pathlib.Path, out_dir: pathlib.Path, stride: int, window: int,
                 k_instr: int, holdout_frac: float, seed: int, min_valid: int):
    task = task_dir.parent.name
    files = sorted((task_dir / "data").glob("episode_*.hdf5"))
    n_hold = max(1, int(round(len(files) * holdout_frac)))
    rng = random.Random(f"{seed}:{task}")
    img_dir = out_dir / "images" / task
    img_dir.mkdir(parents=True, exist_ok=True)

    records, windows, unseen = [], [], {}
    for ep_idx, path in enumerate(files):
        ep = int(path.stem.split("_")[-1])
        state, action, jpegs, instr = load_episode(path)
        ij = task_dir / "instruction" / f"episode_{ep:07d}.json"
        if ij.exists():
            j = json.load(open(ij))
            seen_list = list(dict.fromkeys(j.get("seen", [])))
            unseen[f"{task}/{ep}"] = list(dict.fromkeys(j.get("unseen", [])))
        else:
            seen_list = list(dict.fromkeys(instr or []))
        if not seen_list:
            print(f"[{task}] episode {ep}: no instructions, skipped", file=sys.stderr)
            continue
        split = "holdout" if ep_idx >= len(files) - n_hold else "train"
        for t, w, n in windows_for_episode(state, action, stride, window, min_valid):
            img_name = f"{task}/ep{ep:07d}_t{t:04d}.jpg"
            with open(out_dir / "images" / img_name, "wb") as fh:
                fh.write(jpegs[t])
            picks = rng.sample(seen_list, min(k_instr, len(seen_list)))
            records.append(dict(task=task, episode=ep, t=t, T=len(action), valid=n,
                                image=img_name, split=split, instructions=picks,
                                window_local=len(windows)))
            windows.append(w)
    return task, records, np.stack(windows) if windows else np.zeros((0, window, 14), np.float32), unseen


def _probe_image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.height, im.width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help=".../data/demo_clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task names; default all")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--k-instr", type=int, default=8)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--min-valid", type=int, default=10, help="drop windows with fewer real rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment", default="aloha_agilex")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    root = pathlib.Path(a.data_root)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    task_dirs = sorted(p / a.embodiment for p in root.iterdir() if (p / a.embodiment / "data").is_dir())
    if a.tasks:
        task_dirs = [d for d in task_dirs if d.parent.name in set(a.tasks)]
    print(f"{len(task_dirs)} tasks -> {out}")

    all_records, all_windows, all_unseen = [], [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process_task, d, out, a.stride, a.window, a.k_instr, a.holdout_frac, a.seed, a.min_valid)
                for d in task_dirs]
        for fut in futs:                       # submission order: byte-identical rebuilds
            task, recs, wins, unseen = fut.result()
            offset = sum(len(w) for w in all_windows)
            for r in recs:
                r["action_id"] = offset + r.pop("window_local")
            all_records += recs
            all_windows.append(wins)
            all_unseen.update(unseen)
            print(f"  {task:28s} frames={len(recs):6d}")

    windows = np.concatenate(all_windows)                     # (N, W, 14) with NaN padding
    valid = ~np.isnan(windows[..., 0])
    # Scale statistics from training episodes only (no holdout leakage).
    train_mask = np.zeros(len(windows), bool)
    for r in all_records:
        train_mask[r["action_id"]] = r["split"] == "train"
    flat = windows[train_mask][valid[train_mask]]              # (M, 14)
    q01, q99 = np.nanpercentile(flat, 1, axis=0), np.nanpercentile(flat, 99, axis=0)
    # Arm deltas: symmetric scale so zero motion maps to exactly zero. Grippers are
    # [0, 1] by construction: map to [-1, 1] with a fixed affine.
    scale = np.maximum(np.abs(q01), np.abs(q99))
    scale = np.where(scale > 1e-6, scale, 1.0)
    normed = windows / scale
    normed[..., GRIP] = windows[..., GRIP] * 2.0 - 1.0
    normed[~valid] = PAD
    np.save(out / "actions.npy", normed.astype(np.float16))

    # Instruction table, deduplicated over the whole set.
    table, index = [], {}
    for r in all_records:
        ids = []
        for s in r.pop("instructions"):
            if s not in index:
                index[s] = len(table)
                table.append(s)
            ids.append(index[s])
        r["instruction_ids"] = ids
    json.dump(table, open(out / "instructions.json", "w"), ensure_ascii=False)
    json.dump(all_unseen, open(out / "unseen.json", "w"), ensure_ascii=False)
    with open(out / "samples.jsonl", "w") as fh:
        for r in sorted(all_records, key=lambda r: (r["task"], r["episode"], r["t"])):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(q01=q01.tolist(), q99=q99.tolist(), scale=scale.tolist(), pad=PAD, arm_dims=ARM, grip_dims=GRIP,
                   mapping="arm: (a[t+k] - s[t]) / scale, scale = max(|q01|,|q99|); grip: 2g - 1"),
              open(out / "norm_stats.json", "w"), indent=1)
    n_train = sum(r["split"] == "train" for r in all_records)
    meta = dict(stride=a.stride, window=a.window, action_dim=14, k_instr=a.k_instr, min_valid=a.min_valid,
                holdout_frac=a.holdout_frac, seed=a.seed, tasks=[d.parent.name for d in task_dirs],
                frames=len(all_records), frames_train=n_train, frames_holdout=len(all_records) - n_train,
                pairs=sum(len(r["instruction_ids"]) for r in all_records), instructions=len(table),
                image_size=list(_probe_image_size(out / "images" / all_records[0]["image"])),
                embodiment=a.embodiment, source=str(root))
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
