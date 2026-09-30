#!/usr/bin/env python3
"""Pre-resize every frame once so training does no image work on the CPU.

Writes, next to the dataset:

    images_<S>.u8.npy      (N, S, S, 3) uint8, row i = i-th line of samples.jsonl
    images_<S>.meta.json   size, interpolation, count, a fingerprint of the row order

The resize is the backbone's own (open_clip: bicubic, antialias, straight to SxS),
applied to the decoded JPEG exactly as the online path does, so
(u8 / 255 - mean) / std on the GPU reproduces `preprocess(img)` bit for bit.
The dataset picks the file up automatically (`image_cache: auto`).

Usage: cache_images.py <dataset_dir> [--size 384] [--workers 32]
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

_STATE = {}


def _resize(path, size):
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [size, size], InterpolationMode.BICUBIC, antialias=True)
    return np.asarray(img, dtype=np.uint8)


def _init(root, out, shape, size):
    _STATE.update(root=root, size=size, mm=np.lib.format.open_memmap(out, mode="r+"))
    assert _STATE["mm"].shape == shape


def _work(job):
    start, names = job
    mm, root, size = _STATE["mm"], _STATE["root"], _STATE["size"]
    for k, name in enumerate(names):
        mm[start + k] = _resize(root / "images" / name, size)
    mm.flush()
    return len(names)


def row_fingerprint(names):
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode()); h.update(b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    root = pathlib.Path(a.dataset)
    names = [json.loads(l)["image"] for l in open(root / "samples.jsonl")]
    n, s = len(names), a.size
    out = root / f"images_{s}.u8.npy"
    shape = (n, s, s, 3)
    print(f"{n:,} frames -> {out}  ({n * s * s * 3 / 2**30:.1f} GiB)")
    np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=shape).flush()
    jobs = [(i, names[i:i + a.chunk]) for i in range(0, n, a.chunk)]
    done = 0
    with Pool(a.workers, initializer=_init, initargs=(root, out, shape, s)) as pool:
        for k in pool.imap_unordered(_work, jobs):
            done += k
            if done % (a.chunk * 40) < a.chunk:
                print(f"  {done:,}/{n:,}", flush=True)
    json.dump(dict(size=s, count=n, interpolation="bicubic", antialias=True, dtype="uint8", layout="NHWC",
                   rows="samples.jsonl line order", fingerprint=row_fingerprint(names)),
              open(root / f"images_{s}.meta.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
