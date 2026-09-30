"""Dataset + batch sampler for the verifier training set written by build_dataset.py."""
from __future__ import annotations

import collections
import json
import pathlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VerifierDataset(Dataset):
    """One item = (image tensor, instruction string, window (W, D) float32, frame index).

    Items are (frame, instruction) pairs: frame i with its k-th instruction. `split`
    selects train/holdout frames; `instruction_source` picks 'seen' (the stored ids)
    or 'unseen' (one random unseen sentence per frame, evaluation only).
    """

    def __init__(self, root, split="train", preprocess=None, instruction_source="seen",
                 max_pairs_per_frame=None, seed=0, image_cache="auto", image_size=None):
        self.root = pathlib.Path(root)
        self.preprocess = preprocess
        self.image_cache = self._open_image_cache(image_cache, image_size)
        self.meta = json.load(open(self.root / "meta.json"))
        self.norm = json.load(open(self.root / "norm_stats.json"))
        self.instructions = json.load(open(self.root / "instructions.json"))
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        recs = [json.loads(l) for l in open(self.root / "samples.jsonl")]
        for row, r in enumerate(recs):
            r["row"] = row                                # row in images_<S>.u8.npy
        self.frames = [r for r in recs if split in ("all", r["split"])]
        self.unseen = json.load(open(self.root / "unseen.json")) if instruction_source == "unseen" else None
        rng = random.Random(seed)
        self.pairs = []                                   # (frame_idx, instruction text)
        for fi, r in enumerate(self.frames):
            if instruction_source == "seen":
                ids = r["instruction_ids"][:max_pairs_per_frame] if max_pairs_per_frame else r["instruction_ids"]
                self.pairs += [(fi, self.instructions[i]) for i in ids]
            else:
                cands = self.unseen.get(f"{r['task']}/{r['episode']}") or []
                if cands:
                    self.pairs.append((fi, rng.choice(cands)))
        self.tasks = sorted({r["task"] for r in self.frames})

    def _open_image_cache(self, mode, image_size):
        """'auto' uses images_<S>.u8.npy when it exists and matches the backbone's
        input size; a path forces that file; None/'none' decodes JPEGs online."""
        if mode in (None, "none", False):
            return None
        if mode == "auto":
            if image_size is None:
                return None
            path = self.root / f"images_{image_size}.u8.npy"
            if not path.exists():
                return None
        else:
            path = pathlib.Path(mode)
        arr = np.load(path, mmap_mode="r")
        if image_size is not None and arr.shape[1] != image_size:
            raise ValueError(f"{path} holds {arr.shape[1]}px images, backbone wants {image_size}px")
        return arr

    def __len__(self):
        return len(self.pairs)

    def frame_key(self, pair_idx):
        r = self.frames[self.pairs[pair_idx][0]]
        return r["task"], r["episode"]

    def __getitem__(self, idx):
        fi, text = self.pairs[idx]
        r = self.frames[fi]
        if self.image_cache is not None:
            # uint8 CHW; the model normalises on the GPU, so no CPU image work here
            img = torch.from_numpy(np.array(self.image_cache[r["row"]])).permute(2, 0, 1)
        else:
            img = Image.open(self.root / "images" / r["image"]).convert("RGB")
            if self.preprocess is not None:
                img = self.preprocess(img)
        window = torch.from_numpy(np.asarray(self.actions[r["action_id"]], dtype=np.float32))
        return img, text, window, fi


class EpisodeUniqueBatchSampler:
    """Yields lists of pair indices. Within a batch every (task, episode) appears at
    most once, so in-batch negatives never come from the same demonstration
    (adjacent windows of one episode are near-duplicates and would be false
    negatives). Optionally caps how many pairs each task contributes per epoch so
    long-episode tasks do not dominate.

    Deterministic in (seed, epoch); rank r of world w takes batches[r::w].
    """

    def __init__(self, dataset: VerifierDataset, batch_size: int, seed=0, rank=0, world_size=1,
                 per_task_cap: int | None = None, drop_last=True):
        self.ds, self.bs, self.seed = dataset, batch_size, seed
        self.rank, self.world = rank, world_size
        self.cap, self.drop_last = per_task_cap, drop_last
        self.epoch = 0
        self.by_task = collections.defaultdict(list)
        for i in range(len(dataset)):
            self.by_task[dataset.frame_key(i)[0]].append(i)
        self.keys = [dataset.frame_key(i) for i in range(len(dataset))]
        self._cache = None                                # (epoch, batches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _epoch_indices(self, rng):
        idx = []
        for task, lst in self.by_task.items():
            if self.cap and len(lst) > self.cap:
                idx += rng.sample(lst, self.cap)
            else:
                idx += lst
        rng.shuffle(idx)
        return idx

    def _batches(self):
        if self._cache and self._cache[0] == self.epoch:
            return self._cache[1]
        batches = self._build_batches()
        self._cache = (self.epoch, batches)
        return batches

    def _build_batches(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        pending = collections.deque(self._epoch_indices(rng))
        batches, cur, seen, deferred = [], [], set(), []
        while pending:
            i = pending.popleft()
            k = self.keys[i]
            if k in seen:
                deferred.append(i)
            else:
                cur.append(i); seen.add(k)
            if len(cur) == self.bs:
                batches.append(cur); cur, seen = [], set()
                pending.extendleft(reversed(deferred)); deferred = []
        # leftovers: deferred items that never found a batch, plus the tail
        rest = cur + deferred
        while rest:
            cur, seen, rem = [], set(), []
            for i in rest:
                if self.keys[i] in seen or len(cur) == self.bs:
                    rem.append(i)
                else:
                    cur.append(i); seen.add(self.keys[i])
            if len(cur) == self.bs or (not self.drop_last and cur):
                batches.append(cur)
            if len(rem) == len(rest):          # cannot make progress
                break
            rest = rem
        return batches

    def __iter__(self):
        batches = self._batches()
        n = len(batches) - len(batches) % self.world       # equal count per rank
        for b in batches[self.rank:n:self.world]:
            yield b

    def __len__(self):
        b = len(self._batches())
        return (b - b % self.world) // self.world


def make_collate(tokenizer, context_length):
    def collate(items):
        imgs = torch.stack([it[0] for it in items])
        texts = [it[1] for it in items]
        tokens = tokenizer(texts, context_length=context_length)
        windows = torch.stack([it[2] for it in items])
        frames = torch.tensor([it[3] for it in items])
        return imgs, tokens, windows, frames, texts
    return collate
