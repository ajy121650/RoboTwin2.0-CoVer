#!/usr/bin/env python3
"""Offline retrieval evaluation of a verifier checkpoint on the holdout episodes.

For every holdout frame we embed (image, instruction) and its expert window, then
ask: among a pool of windows, does the frame's own window score highest?  Three
pools of increasing difficulty are reported, for seen and unseen instructions:

  global    all holdout windows of all tasks       (what CoVer's top-k measures)
  task      holdout windows of the same task       (other episodes, other phases)
  episode   windows of the same episode only        (other phases of the same demo:
                                                     the hardest, closest to ranking
                                                     policy candidates)

Usage: eval_offline.py <ckpt.pt> [--data DIR] [--batch 256] [--max-frames N] [--out report.json]
"""
from __future__ import annotations

import argparse, collections, json, pathlib, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verifier.model import VerifierConfig, build            # noqa: E402
from verifier.data import VerifierDataset, make_collate     # noqa: E402


def load_checkpoint(path, device, hf_home=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = VerifierConfig(**ck["model_config"])
    model, preprocess, tokenizer = build(mcfg, hf_home=hf_home or ck["config"].get("hf_home"))
    model.load_head_state_dict(ck["head_state_dict"])
    model.to(device).eval()
    return model, preprocess, tokenizer, ck


@torch.no_grad()
def embed(model, loader, device):
    fs, as_, frames = [], [], []
    for imgs, tokens, windows, fi, _ in loader:
        f, a = model(imgs.to(device), tokens.to(device), windows.to(device))
        fs.append(f.cpu()); as_.append(a.cpu()); frames.append(fi)
    return torch.cat(fs), torch.cat(as_), torch.cat(frames)


def evaluate(model, ds, device, batch, tokenizer, ctx_len, label, workers=8):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=make_collate(tokenizer, ctx_len))
    f, a, frames = embed(model, loader, device)
    frames = frames.numpy()
    tasks = np.array([ds.frames[i]["task"] for i in frames])
    eps = np.array([f"{ds.frames[i]['task']}/{ds.frames[i]['episode']}" for i in frames])
    S = (f @ a.t()).numpy()                                      # (N, N) context x window
    n = len(f)
    out = {}
    for pool_name, pool_of in (("global", lambda i: np.arange(n)),
                               ("task", lambda i: np.where(tasks == tasks[i])[0]),
                               ("episode", lambda i: np.where(eps == eps[i])[0])):
        ranks, sizes, ties, per_task = [], [], 0, collections.defaultdict(list)
        for i in range(n):
            p = pool_of(i)
            row, t = S[i, p], S[i, i]
            # Pessimistic rank: ties with the target count against it, so a constant
            # scorer lands at chance rather than at 100%.
            n_tie = int((row == t).sum()) - 1
            r = int((row > t).sum()) + n_tie
            ties += n_tie > 0
            ranks.append(r); sizes.append(len(p)); per_task[tasks[i]].append(r)
        ranks, sizes = np.array(ranks), np.array(sizes)
        out[pool_name] = dict(top1=float((ranks == 0).mean()), top5=float((ranks < 5).mean()),
                              mean_rank=float(ranks.mean()), mean_pool=float(sizes.mean()),
                              chance_top1=float((1 / sizes).mean()), queries_with_ties=int(ties),
                              per_task_top1={t: float((np.array(v) == 0).mean()) for t, v in sorted(per_task.items())})
    print(f"[{label}] frames {n}")
    for k, v in out.items():
        print(f"  {k:8s} top1 {100*v['top1']:5.1f}%  top5 {100*v['top5']:5.1f}%  mean rank {v['mean_rank']:6.1f}  "
              f"pool {v['mean_pool']:7.1f}  chance {100*v['chance_top1']:.2f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--data"); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--hf-home", default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer, ck = load_checkpoint(a.ckpt, device, a.hf_home)
    data = a.data or ck["config"]["data"]
    ctx_len = model.clip.context_length
    report = dict(ckpt=a.ckpt, step=ck.get("step"), data=data)
    for src in ("seen", "unseen"):
        ds = VerifierDataset(data, "holdout", preprocess, instruction_source=src, max_pairs_per_frame=1,
                             image_cache=ck["config"].get("image_cache", "auto"), image_size=model.image_size)
        if len(ds) == 0:
            print(f"[{src}] no holdout pairs (empty unseen.json?), skipped"); continue
        if a.max_frames and len(ds) > a.max_frames:
            rng = np.random.default_rng(0)
            keep = sorted(rng.choice(len(ds), a.max_frames, replace=False).tolist())
            ds.pairs = [ds.pairs[i] for i in keep]
        report[src] = evaluate(model, ds, device, a.batch, tokenizer, ctx_len, src, workers=a.workers)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
