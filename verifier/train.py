#!/usr/bin/env python3
"""Train the verifier with symmetric InfoNCE (CoVer Level 0).

Single GPU:   python train.py --config configs/default.yaml
Multi GPU:    torchrun --nproc_per_node 8 train.py --config configs/default.yaml
Overrides:    --data DIR --out DIR --set train.lr=3e-5 --set train.max_steps=20
"""
from __future__ import annotations

import argparse, dataclasses, json, math, os, pathlib, shutil, sys, time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verifier.model import VerifierConfig, build, load_cover_warm_start   # noqa: E402
from verifier.data import VerifierDataset, EpisodeUniqueBatchSampler, make_collate  # noqa: E402


def set_nested(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    old = d.get(keys[-1])
    d[keys[-1]] = yaml.safe_load(value) if not isinstance(old, str) else value


def info_nce(f, a, logit_scale, rank, world):
    """Symmetric InfoNCE over embeddings gathered from every rank. Gradients flow
    to the local rows only (the standard CLIP recipe)."""
    if world > 1:
        f_all = torch.cat([g if i == rank else g.detach() for i, g in enumerate(all_gather_with_grad(f))])
        a_all = torch.cat([g if i == rank else g.detach() for i, g in enumerate(all_gather_with_grad(a))])
    else:
        f_all, a_all = f, a
    logits = logit_scale * f_all @ a_all.t()
    labels = torch.arange(len(f_all), device=f.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    with torch.no_grad():
        top1 = (logits.argmax(1) == labels).float().mean()
        top5 = (logits.topk(min(5, len(f_all)), dim=1).indices == labels[:, None]).any(1).float().mean()
    return loss, top1, top5, len(f_all)


def all_gather_with_grad(t):
    out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    out[dist.get_rank()] = t
    return out


@torch.no_grad()
def validate(model, loader, device, logit_scale):
    model.eval()
    fs, as_ = [], []
    for imgs, tokens, windows, _, _ in loader:
        f, a = model(imgs.to(device, non_blocking=True), tokens.to(device), windows.to(device))
        fs.append(f); as_.append(a)
    f, a = torch.cat(fs), torch.cat(as_)
    logits = logit_scale * f @ a.t()
    labels = torch.arange(len(f), device=device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    top1 = (logits.argmax(1) == labels).float().mean()
    top5 = (logits.topk(min(5, len(f)), dim=1).indices == labels[:, None]).any(1).float().mean()
    model.train(); model.clip.eval()
    return dict(val_loss=loss.item(), val_top1=top1.item(), val_top5=top5.item(), val_n=len(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(pathlib.Path(__file__).parent / "configs/default.yaml"))
    ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.data: cfg["data"] = args.data
    if args.out: cfg["out"] = args.out
    for s in args.set:
        k, v = s.split("=", 1); set_nested(cfg, k, v)
    if not cfg.get("data") or not cfg.get("out"):
        ap.error("--data and --out are required (or set data/out in the config)")
    tr = cfg["train"]

    # -- distributed
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0
    out = pathlib.Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
    if is_main:
        yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
        shutil.copy(pathlib.Path(cfg["data"]) / "norm_stats.json", out / "norm_stats.json")
        shutil.copy(pathlib.Path(cfg["data"]) / "meta.json", out / "data_meta.json")
    torch.manual_seed(tr["seed"] + rank)

    # -- model (rank 0 fetches the backbone first so eight ranks do not race the download)
    mcfg = VerifierConfig(**cfg["model"])
    if ddp and not is_main:
        dist.barrier()
    model, preprocess, tokenizer = build(mcfg, hf_home=cfg.get("hf_home"))
    if ddp and is_main:
        dist.barrier()
    if cfg.get("warm_start"):
        loaded = load_cover_warm_start(model, cfg["warm_start"])
        if is_main: print(f"warm start from {cfg['warm_start']}: {loaded}")
    model.to(device)
    ctx_len = model.clip.context_length
    params = model.trainable_parameters()
    n_params = sum(p.numel() for p in params)
    if is_main: print(f"trainable params: {n_params/1e6:.1f}M | device {device} | world {world}")
    ddp_model = (torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
        if ddp else model)

    # -- data
    train_ds = VerifierDataset(cfg["data"], "train", preprocess)
    sampler = EpisodeUniqueBatchSampler(train_ds, tr["batch_size"], seed=tr["seed"], rank=rank, world_size=world,
                                        per_task_cap=tr.get("per_task_cap"))
    collate = make_collate(tokenizer, ctx_len)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=tr["num_workers"], collate_fn=collate,
                              pin_memory=True, persistent_workers=tr["num_workers"] > 0, prefetch_factor=4 if tr["num_workers"] > 0 else None)
    val_ds = VerifierDataset(cfg["data"], "holdout", preprocess, max_pairs_per_frame=1)
    g = torch.Generator().manual_seed(tr["seed"])
    val_idx = torch.randperm(len(val_ds), generator=g)[: tr["val_pairs"]].tolist()
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=tr["val_batch_size"], shuffle=False,
                            num_workers=tr["num_workers"] // 2, collate_fn=collate, pin_memory=True)
    steps_per_epoch = len(sampler)
    total_steps = tr["max_steps"] or steps_per_epoch * tr["epochs"]
    if is_main:
        print(f"train pairs {len(train_ds):,} | frames {len(train_ds.frames):,} | tasks {len(train_ds.tasks)} | "
              f"steps/epoch {steps_per_epoch} (per rank) | total steps {total_steps} | val pairs {len(val_idx)}")

    # -- optim
    decay = [p for p in params if p.ndim >= 2 and p.numel() > 1]
    no_decay = [p for p in params if not (p.ndim >= 2 and p.numel() > 1)]   # biases, norms, scalars, queries
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=tr["weight_decay"]),
                             dict(params=no_decay, weight_decay=0.0)], lr=tr["lr"], betas=(0.9, 0.98))
    warm = tr["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total_steps - warm))))))
    step, epoch, best, batch_in_epoch = 0, 0, -1.0, 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_head_state_dict(ck["head_state_dict"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        step, epoch, best = ck["step"], ck["epoch"], ck.get("best_val", best)
        batch_in_epoch = ck.get("batch_in_epoch", 0)
        if is_main: print(f"resumed at step {step} epoch {epoch} (skipping {batch_in_epoch} batches of this epoch)")

    log_f = open(out / "log.jsonl", "a") if is_main else None
    use_wandb = tr.get("wandb") and is_main
    if use_wandb:
        import wandb; wandb.init(project=tr["wandb_project"], name=out.name, id=out.name, config=cfg, resume="allow")

    def save(name, extra=None):
        if not is_main: return
        torch.save(dict(head_state_dict=model.head_state_dict(), model_config=dataclasses.asdict(mcfg), config=cfg,
                        norm_stats=json.load(open(out / "norm_stats.json")), data_meta=json.load(open(out / "data_meta.json")),
                        step=step, epoch=epoch, batch_in_epoch=batch_in_epoch, best_val=best,
                        optimizer=opt.state_dict(), scheduler=sched.state_dict(), **(extra or {})), out / name)

    model.train(); model.clip.eval()
    t0 = time.time(); done = step >= total_steps
    while not done:
        sampler.set_epoch(epoch)
        skip = batch_in_epoch                       # only non-zero right after a resume
        for imgs, tokens, windows, _, _ in train_loader:
            if skip:
                skip -= 1
                continue
            batch_in_epoch += 1
            imgs = imgs.to(device, non_blocking=True); tokens = tokens.to(device); windows = windows.to(device)
            f, a = ddp_model(imgs, tokens, windows)
            scale = model.logit_scale.clamp(max=math.log(100)).exp()
            loss, top1, top5, n_all = info_nce(f, a, scale, rank, world)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"])
            opt.step(); sched.step(); step += 1

            if is_main and step % tr["log_every"] == 0:
                rec = dict(step=step, epoch=epoch, loss=loss.item(), top1=top1.item(), top5=top5.item(), n=n_all,
                           lr=sched.get_last_lr()[0], scale=scale.item(), sec=time.time() - t0)
                print(json.dumps(rec)); log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if use_wandb: wandb.log(rec, step=step)
            if step % tr["val_every"] == 0 or step == total_steps:
                v = validate(model, val_loader, device, scale) if is_main else None
                if is_main:
                    v.update(step=step, epoch=epoch); print(json.dumps(v)); log_f.write(json.dumps(v) + "\n"); log_f.flush()
                    if use_wandb: wandb.log(v, step=step)
                    if v["val_top1"] > best:                 # val_loss drifts with logit_scale; top-1 does not
                        best = v["val_top1"]; save("best.pt", dict(val=v))
                    save("last.pt", dict(val=v))
                if ddp: dist.barrier()
            if step >= total_steps:
                done = True; break
        if done:
            break                      # stopped by max_steps mid-epoch: no epoch checkpoint
        epoch += 1; batch_in_epoch = 0
        save(f"epoch{epoch:03d}.pt")
    save("last.pt")
    if is_main: print(f"done: {step} steps, best val top1 {best:.4f}, {time.time()-t0:.0f}s")
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
