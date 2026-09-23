#!/usr/bin/env python3
"""Sanity report for a built verifier dataset, plus a contact sheet of a few samples.

Usage: inspect_dataset.py <dataset_dir> [--n-vis 6] [--out contact.png]
"""
import argparse, collections, json, pathlib, random
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n-vis", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = pathlib.Path(a.dataset)
    meta = json.load(open(d / "meta.json"))
    stats = json.load(open(d / "norm_stats.json"))
    instr = json.load(open(d / "instructions.json"))
    recs = [json.loads(l) for l in open(d / "samples.jsonl")]
    acts = np.load(d / "actions.npy", mmap_mode="r")

    print(f"# {d.name}")
    print(f"tasks {len(meta['tasks'])} | frames {meta['frames']:,} (train {meta['frames_train']:,} / holdout {meta['frames_holdout']:,})"
          f" | pairs {meta['pairs']:,} | unique instructions {meta['instructions']:,}")
    print(f"stride {meta['stride']} window {meta['window']} k_instr {meta['k_instr']} min_valid {meta.get('min_valid')}")
    print("scale:", np.round(stats["scale"], 3).tolist())

    per_task = collections.Counter(r["task"] for r in recs)
    per_task_hold = collections.Counter(r["task"] for r in recs if r["split"] == "holdout")
    print("\nframes per task (train+holdout / holdout):")
    for t in meta["tasks"]:
        print(f"  {t:28s} {per_task[t]:6d} / {per_task_hold[t]:4d}")
    k = np.array([len(r["instruction_ids"]) for r in recs]); print(f"\ninstructions per frame: min {k.min()} mean {k.mean():.2f} max {k.max()}")
    lens = [len(s.split()) for s in instr]; print(f"instruction words: min {min(lens)} mean {np.mean(lens):.1f} max {max(lens)}")
    ep_per_task = collections.defaultdict(set)
    for r in recs: ep_per_task[r["task"]].add(r["episode"])
    print("episodes per task:", sorted({len(v) for v in ep_per_task.values()}))

    # action checks on a sample of windows
    idx = np.random.default_rng(a.seed).choice(len(acts), min(5000, len(acts)), replace=False)
    w = acts[np.sort(idx)].astype(np.float32)
    valid = w[..., 0] != stats["pad"]
    print(f"\nwindows sampled {len(w)} | valid rows {valid.mean():.3f} | value range {w[valid].min():.2f}..{w[valid].max():.2f}")
    print("k=0 mean |arm delta| (should be ~0):", np.abs(w[:, 0][valid[:, 0]][:, stats["arm_dims"]]).mean().round(4))
    print("k=last-valid mean |arm delta|:", np.abs(w[:, -1][valid[:, -1]][:, stats["arm_dims"]]).mean().round(3))
    g = w[..., stats["grip_dims"]][valid]; print("gripper values in [-1,1]:", bool(g.min() >= -1 - 1e-3 and g.max() <= 1 + 1e-3))
    frac_out = (np.abs(w[valid][:, stats["arm_dims"]]) > 1).mean(); print(f"arm values beyond |1| (expected ~2%): {100*frac_out:.2f}%")

    if a.out:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        rng = random.Random(a.seed)
        picks = rng.sample(recs, a.n_vis)
        fig, axes = plt.subplots(a.n_vis, 2, figsize=(11, 2.6 * a.n_vis))
        for i, r in enumerate(picks):
            axes[i, 0].imshow(Image.open(d / "images" / r["image"])); axes[i, 0].axis("off")
            axes[i, 0].set_title(f"{r['task']} ep{r['episode']} t={r['t']}/{r['T']} [{r['split']}]", fontsize=8)
            ww = acts[r["action_id"]].astype(np.float32); v = ww[:, 0] != stats["pad"]
            for dd in stats["arm_dims"]: axes[i, 1].plot(np.where(v)[0], ww[v, dd], lw=0.8)
            for dd in stats["grip_dims"]: axes[i, 1].plot(np.where(v)[0], ww[v, dd], "k--", lw=0.8)
            axes[i, 1].set_ylim(-1.3, 1.3); axes[i, 1].set_xlim(0, ww.shape[0]); axes[i, 1].grid(alpha=0.3)
            axes[i, 1].set_title(instr[r["instruction_ids"][0]][:90], fontsize=8)
        plt.tight_layout(); plt.savefig(a.out, dpi=110); print("wrote", a.out)


if __name__ == "__main__":
    main()
