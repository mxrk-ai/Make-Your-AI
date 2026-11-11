"""The diffusion model: starts from noise and denoises towards an adapter latent that fits a description.

Small transformer over the adapter's tokens (one per targeted module). Each block: attention between
tokens, attention to the description's clause embeddings, then a small MLP. It predicts the clean
latent; sampling is DDIM with classifier-free guidance.
"""

from __future__ import annotations

import math

import torch
from torch import nn

T_STEPS = 1000


def alpha_bar(t: torch.Tensor) -> torch.Tensor:
    """Cosine noise schedule. t in [0, 1]; 1 = pure noise."""
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def add_noise(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
    noise = torch.randn_like(x0) if noise is None else noise
    ab = alpha_bar(t).view(-1, 1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cond, cond_pad):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.n2(x), cond, cond, key_padding_mask=cond_pad, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class Denoiser(nn.Module):
    def __init__(self, n_tokens: int, latent_dim: int, cond_dim: int = 384, d: int = 256, layers: int = 4, heads: int = 4):
        super().__init__()
        self.cfg = dict(n_tokens=n_tokens, latent_dim=latent_dim, cond_dim=cond_dim, d=d, layers=layers, heads=heads)
        self.inp = nn.Linear(latent_dim, d)
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.cond = nn.Linear(cond_dim, d)
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)  # stands in for "no description"
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, latent_dim))

    def forward(self, xt, t, cond, mask):
        """xt [B, tokens, latent]; t [B] in [0,1]; cond [B, clauses, cond_dim]; mask [B, clauses] True = real clause."""
        d = self.pos.shape[1]
        freqs = torch.exp(-math.log(10000) * torch.arange(d // 2, device=xt.device) / (d // 2))
        ang = t.view(-1, 1) * 1000 * freqs
        temb = self.time(torch.cat([ang.sin(), ang.cos()], dim=-1)).unsqueeze(1)
        c = torch.cat([self.null.expand(len(xt), -1, -1), self.cond(cond)], dim=1)  # null token is always visible
        pad = torch.cat([torch.zeros(len(xt), 1, dtype=torch.bool, device=xt.device), ~mask], dim=1)
        h = self.inp(xt) + self.pos + temb
        for b in self.blocks:
            h = b(h, c, pad)
        return self.out(h)


def train_denoiser(model: Denoiser, latents: torch.Tensor, cond_fn, steps: int = 4000, batch_size: int = 64,
                   lr: float = 3e-4, p_uncond: float = 0.1, device: str = "cpu", seed: int = 0, log=print) -> list[float]:
    """latents [N, tokens, dim]. cond_fn(indices) -> (cond, mask) for those adapters (fresh phrasing each call)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (0.5 * (1 + math.cos(math.pi * s / steps))))
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(latents), (batch_size,), generator=g)
        x0 = latents[idx].to(device)
        cond, mask = cond_fn(idx.tolist())
        mask = mask & (torch.rand(batch_size, 1, generator=g) > p_uncond)  # sometimes hide the description
        t = torch.rand(batch_size, generator=g).to(device)
        xt, _ = add_noise(x0, t)
        loss = ((model(xt, t, cond.to(device), mask.to(device)) - x0) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        losses.append(loss.item())
        if (step + 1) % max(steps // 10, 1) == 0:
            log(f"  generator step {step + 1}/{steps}  loss {sum(losses[-100:]) / len(losses[-100:]):.4f}")
    model.eval()
    return losses


@torch.no_grad()
def sample(model: Denoiser, cond, mask, steps: int = 50, guidance: float = 2.0, seed: int = 0,
           start: torch.Tensor | None = None, start_t: float = 1.0) -> torch.Tensor:
    """DDIM sampling. With `start` and `start_t` < 1 it re-noises an existing latent part of the way and
    denoises it again (used to pull an edited adapter back towards realistic ones)."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    n = len(cond)
    shape = (n, model.cfg["n_tokens"], model.cfg["latent_dim"])
    noise = torch.randn(shape, generator=g).to(dev)
    cond, mask = cond.to(dev), mask.to(dev)
    ts = torch.linspace(start_t, 0, steps + 1).to(dev)
    x = noise if start is None else add_noise(start.to(dev), ts[:1].expand(n), noise)[0]
    for i in range(steps):
        t, t_next = ts[i].expand(n), ts[i + 1].expand(n)
        x0 = model(x, t, cond, mask)
        if guidance != 1.0:
            x0_un = model(x, t, cond, torch.zeros_like(mask))
            x0 = x0_un + guidance * (x0 - x0_un)
        ab, ab_next = alpha_bar(t).view(-1, 1, 1), alpha_bar(t_next).view(-1, 1, 1)
        eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-4)
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x.cpu()
