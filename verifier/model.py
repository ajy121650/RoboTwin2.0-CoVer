"""Verifier = frozen SigLIP2 (image, text) tower + trainable heads, CLIP-style.

Ported from CoVer's VLA_SigLIP2_Bridge (bridge_verifier/ensemble_eval/{model,
finetune_trajectory_bridge_ddp}.py) with three deliberate changes:
  * action window is (W=50, D=14) joint-space deltas instead of (10, 7) EEF deltas
  * the text padding mask is honoured in text pooling and text-aware attention
    (CoVer let pad tokens attend; keep `text_mask: false` to reproduce that)
  * no ensemble machinery here; an ensemble is N checkpoints averaged at inference

Score(image, text, window) = cosine(f, a) with f = context embedding, a = action
embedding, both L2-normalised. Training uses symmetric InfoNCE on f·aᵀ.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- heads

class CrossAttentionBlock(nn.Module):
    def __init__(self, kv_dim: int, q_dim: int, mlp_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(q_dim, num_heads, batch_first=True, kdim=kv_dim, vdim=kv_dim)
        self.mlp = nn.Sequential(nn.Linear(q_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, q_dim))
        self.q_layer_norm = nn.LayerNorm(q_dim)
        self.layer_norm = nn.LayerNorm(q_dim)

    def forward(self, q, kv, key_padding_mask: Optional[torch.Tensor] = None):
        q = self.q_layer_norm(q)
        attn, _ = self.attention(q, kv, kv, key_padding_mask=key_padding_mask)
        q = q + attn
        q = self.layer_norm(q)
        return q + self.mlp(q)


def sincos_position_embedding(seq_len: int, dim: int) -> torch.Tensor:
    pos = torch.arange(seq_len).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.einsum("i,j->ij", pos, inv_freq)
    return torch.cat((ang.sin(), ang.cos()), dim=-1)


class TextAwareVisualExtraction(nn.Module):
    """Each text token attends over image patches (ClearCLIP-style), giving
    instruction-conditioned visual tokens."""

    def __init__(self, num_patches: int, vision_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.register_buffer("pos_emb", sincos_position_embedding(num_patches, vision_dim))

    def forward(self, patches, text_tokens, value_scale: float = 1.0):
        """patches/text_tokens are unit-norm; the attention logits are cosines / T.
        `value_scale` rescales the patch content that gets pooled (see
        VerifierConfig.token_scale) without touching the logits."""
        sim = torch.einsum("bij,bkj->bik", text_tokens, patches)           # (B, T, P) cosines
        attn = F.softmax(sim / self.temperature.clamp(1e-3, 100), dim=-1)
        values = patches * value_scale + self.pos_emb
        return torch.einsum("bik,bkj->bij", attn, values)                   # (B, T, Dv)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=2, num_readouts=1):
        super().__init__()
        assert output_dim % num_readouts == 0
        self.num_readouts = num_readouts
        d = output_dim // num_readouts
        self.query = nn.Parameter(torch.randn(1, num_readouts, d) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(input_dim, d, output_dim, num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        q = self.query.expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            q = blk(q, x, key_padding_mask)
        return self.layer_norm(q).reshape(x.shape[0], -1)


class ActionTransformerEncoder(nn.Module):
    """(B, W, D) padded windows -> (B, E). Padding rows carry `pad_value` in dim 0."""

    def __init__(self, action_dim, embed_dim, num_layers=4, num_heads=8, dropout=0.1, pad_value=-5.0, window=50,
                 pos_emb=True):
        super().__init__()
        self.pad_value = pad_value
        self.step_encoder = nn.Linear(action_dim, embed_dim)
        # CoVer's encoder has no position embedding: with masked mean pooling it is
        # permutation-invariant over the window. Keep pos_emb=False to reproduce that.
        pe = sincos_position_embedding(window, embed_dim) if pos_emb else torch.zeros(window, embed_dim)
        self.register_buffer("pos_emb", pe)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 2,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, windows):
        pad = (windows == self.pad_value).all(-1)                        # (B, W) True = padding
        x = self.step_encoder(windows) + self.pos_emb[: windows.shape[1]]
        x = self.encoder(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        return (x * keep).sum(1) / keep.sum(1).clamp_min(1.0)


class ActionMLPEncoder(nn.Module):
    def __init__(self, action_dim, embed_dim, window=50, pad_value=-5.0):
        super().__init__()
        self.pad_value = pad_value
        self.net = nn.Sequential(nn.Linear(window * action_dim, 512), nn.LayerNorm(512), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(512, embed_dim))

    def forward(self, windows):
        pad = (windows == self.pad_value).all(-1, keepdim=True)
        w = torch.where(pad, torch.zeros_like(windows), windows)
        return self.net(w.flatten(1))


# ----------------------------------------------------------------------------- model

@dataclasses.dataclass
class VerifierConfig:
    backbone: str = "hf-hub:timm/ViT-L-16-SigLIP2-384"
    text_pool_dim: int = 512
    vision_pool_dim: int = 512
    pool_heads: int = 8
    pool_layers: int = 4
    num_readouts: int = 1
    action_dim: int = 14
    window: int = 50
    action_encoder: str = "transformer"       # or "mlp"
    action_layers: int = 4
    action_pos_emb: bool = True               # False = CoVer (no position information in the window)
    action_dropout: float = 0.1
    pad_value: float = -5.0
    text_mask: bool = True
    # Token features leave the frozen towers L2-normalised (per-dim std ~0.03). CoVer
    # feeds them as-is, which at init makes the pooled context nearly input-independent
    # (the learned query and pos_emb are ~1 per dim). "sqrt_dim" rescales the tokens
    # that enter the pooling heads to unit per-dim scale; the text-aware attention
    # logits stay cosines so its temperature keeps a gradient. "none" reproduces
    # CoVer exactly (required with a CoVer warm start).
    token_scale: str = "sqrt_dim"
    logit_scale_init: float = 2.6592          # ln(1/0.07)


class Verifier(nn.Module):
    def __init__(self, cfg: VerifierConfig, clip_model, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.to(torch.bfloat16).eval()

        trunk = self.clip.visual.trunk
        vision_dim = trunk.num_features
        patch = trunk.patch_embed.proj.kernel_size[0]
        img = self.clip.visual.image_size[0] if hasattr(self.clip.visual, "image_size") else 384
        self.num_patches = (img // patch) ** 2
        text_dim = self.clip.text.output_dim

        self.text_aware = TextAwareVisualExtraction(self.num_patches, vision_dim)
        self.text_pool = AttentionPooling(text_dim, cfg.text_pool_dim, cfg.pool_heads, cfg.pool_layers, cfg.num_readouts)
        self.vision_pool = AttentionPooling(vision_dim, cfg.vision_pool_dim, cfg.pool_heads, cfg.pool_layers, cfg.num_readouts)
        self.context_proj = nn.Linear(cfg.text_pool_dim + cfg.vision_pool_dim, cfg.vision_pool_dim)
        if cfg.action_encoder == "transformer":
            self.action_encoder = ActionTransformerEncoder(cfg.action_dim, cfg.vision_pool_dim, cfg.action_layers,
                                                           cfg.pool_heads, cfg.action_dropout, cfg.pad_value, cfg.window,
                                                           pos_emb=cfg.action_pos_emb)
        else:
            self.action_encoder = ActionMLPEncoder(cfg.action_dim, cfg.vision_pool_dim, cfg.window, cfg.pad_value)
        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale_init))

        # Token-level features come out of forward hooks: the last attention output of
        # the vision trunk (ClearCLIP) and the text transformer's hidden states.
        self._acts = {}
        trunk.blocks[-1].attn.register_forward_hook(lambda m, i, o: self._acts.__setitem__("img", o))
        self.clip.text.transformer.register_forward_hook(lambda m, i, o: self._acts.__setitem__("txt", o))

    # -- frozen features ------------------------------------------------------------
    @torch.no_grad()
    def _token_features(self, images, tokens):
        self.clip.eval()
        if images.dtype == torch.uint8:                # pre-resized cache: normalise here, on the GPU
            mean = torch.tensor(getattr(self.clip.visual, "image_mean", (0.5, 0.5, 0.5)), device=images.device)
            std = torch.tensor(getattr(self.clip.visual, "image_std", (0.5, 0.5, 0.5)), device=images.device)
            images = (images.float() / 255.0 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        self.clip.encode_image(images.to(torch.bfloat16), normalize=False)
        self.clip.encode_text(tokens, normalize=False)
        patches = self._acts["img"]                 # (B, 576, 1024)
        if patches.shape[1] == self.num_patches + 1:
            patches = patches[:, 1:]
        txt = self._acts["txt"]                     # (B, 64, 1024)
        txt = self.clip.text.ln_final(txt)
        if getattr(self.clip.text, "text_projection", None) is not None:
            txt = self.clip.text.text_projection(txt)
        return F.normalize(patches.float(), dim=-1), F.normalize(txt.float(), dim=-1)

    @property
    def token_value_scale(self) -> float:
        return float(self.clip.text.output_dim) ** 0.5 if self.cfg.token_scale == "sqrt_dim" else 1.0

    # -- embeddings -----------------------------------------------------------------
    def encode_context(self, images, tokens):
        patches, txt = self._token_features(images, tokens)
        mask = (tokens == self.pad_id) if self.cfg.text_mask else None
        s = self.token_value_scale
        vis_tokens = self.text_aware(patches, txt, value_scale=s)      # (B, T, Dv)
        vis = self.vision_pool(vis_tokens, mask)
        tx = self.text_pool(txt * s, mask)
        f = self.context_proj(torch.cat([tx, vis], dim=-1))
        return F.normalize(f, dim=-1)

    def encode_actions(self, windows):
        return F.normalize(self.action_encoder(windows.float()), dim=-1)

    def forward(self, images, tokens, windows):
        return self.encode_context(images, tokens), self.encode_actions(windows)

    @property
    def image_size(self) -> int:
        size = getattr(self.clip.visual, "image_size", 384)
        return size[0] if isinstance(size, (tuple, list)) else size

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("clip.")]

    def head_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("clip.")}

    def load_head_state_dict(self, sd, strict=True):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("clip.")]
        if strict and (missing or unexpected):
            raise RuntimeError(f"head state mismatch: missing={missing} unexpected={unexpected}")
        return missing, unexpected


def build(cfg: VerifierConfig, hf_home: Optional[str] = None):
    """Returns (model, image_preprocess, tokenizer). Backbone weights come from the
    HF hub; `hf_home` (or the HF_HOME env var, which wins) points the cache at a
    disk with room for the ~3.3 GB SigLIP2-L download."""
    import os

    if hf_home:
        os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import open_clip
    
    clip_model, preprocess = open_clip.create_model_from_pretrained(cfg.backbone)
    tokenizer = open_clip.get_tokenizer(cfg.backbone)
    pad_id = getattr(clip_model.text, "pad_id", 0)
    return Verifier(cfg, clip_model, pad_id=pad_id), preprocess, tokenizer


def load_cover_warm_start(model: Verifier, path: str) -> list[str]:
    """Copy the text/vision heads from a CoVer bridge checkpoint (ensemble_components[0]).
    The action encoder is left untouched (7-D vs 14-D)."""
    import warnings
    ck = torch.load(path, map_location="cpu", weights_only=False)
    comp = ck["ensemble_components"][0] if "ensemble_components" in ck else ck.get("model_state_dict", ck)
    mapping = {"text_aware_visual_extraction": "text_aware", "text_pooling": "text_pool",
               "vision_poolings": "vision_pool", "input_projection": "context_proj"}
    if model.cfg.token_scale != "none" or model.cfg.text_mask:
        warnings.warn("warm start from CoVer with token_scale != 'none' or text_mask=true: the heads were "
                      "trained under CoVer's scaling; set model.token_scale=none and model.text_mask=false "
                      "to reproduce it.")
    loaded = []
    for src, dst in mapping.items():
        if src in comp and isinstance(comp[src], dict):
            sd = comp[src]                                          # nested per-module dicts
        else:
            sd = {k[len(src) + 1:]: v for k, v in comp.items() if k.startswith(src + ".")}   # flat
            if not sd:
                continue
        sub = getattr(model, dst)
        # CoVer's CrossAttentionBlock used timm Mlp (fc1/fc2); ours is Sequential(0, 2).
        sd = {k.replace("mlp.fc1", "mlp.0").replace("mlp.fc2", "mlp.2"): v for k, v in sd.items()}
        missing, unexpected = sub.load_state_dict(sd, strict=False)
        if unexpected or [m for m in missing if "pos_emb" not in m]:
            raise RuntimeError(f"{src}: missing={missing} unexpected={unexpected}")
        loaded.append(dst)
    if not loaded:
        raise RuntimeError(f"warm start: no matching heads found in {path}")
    return loaded
