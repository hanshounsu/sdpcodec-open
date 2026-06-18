# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532

from collections import namedtuple
from functools import wraps
import os

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from packaging import version
from torch import einsum, nn


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (
            use_flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu
        self.config = namedtuple(
            "EfficientAttentionConfig",
            ["enable_flash", "enable_math", "enable_mem_efficient"],
        )
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None
        self._allow_flash_non_a100 = os.environ.get("PERCEIVER_FLASH_ALLOW_NON_A100", "0") == "1"

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(True, False, False)
        elif self._allow_flash_non_a100 and device_properties.major >= 8:
            print_once(
                "Non-A100 SM>=80 GPU detected, allowing flash attention when supported"
            )
            # Allow flash + fallback kernels for broader compatibility.
            self.cuda_config = self.config(True, True, True)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

        return out

    def forward(self, q, k, v, mask=None, return_attn=False, attn_postprocess=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash and not return_attn and attn_postprocess is None:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn_probs = sim.softmax(dim=-1)
        aux = None
        if attn_postprocess is not None:
            processed = attn_postprocess(attn_probs)
            if isinstance(processed, tuple):
                attn_probs, aux = processed
            else:
                attn_probs = processed

        attn = self.attn_dropout(attn_probs)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        if return_attn:
            return out, attn_probs, aux
        return out


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, "b d -> b 1 d"), (gamma, beta))
        return out * gamma + beta


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.Sequential(
            Rearrange("b n d -> b d n"),
            CausalConv1d(dim_inner, dim_inner, 3),
            Rearrange("b d n -> b n d"),
        )

    return Sequential(
        nn.Linear(dim, dim_inner * 2), GEGLU(), conv, nn.Linear(dim_inner, dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None, return_attn=False, attn_postprocess=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(
            q,
            k,
            v,
            mask=mask,
            return_attn=return_attn,
            attn_postprocess=attn_postprocess,
        )

        attn_probs = None
        aux = None
        if return_attn:
            out, attn_probs, aux = out

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if return_attn:
            return out, attn_probs, aux
        return out


class AttentionPatternQuantizer(nn.Module):
    """
    Quantize a full attention distribution over memory slots to one of K learned
    prototype patterns. This matches a fixed small-bit budget much better than
    quantizing each scalar attention value independently.
    """

    def __init__(self, codebook_size: int, num_bins: int, share_across_heads: bool = True):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.num_bins = int(num_bins)
        self.share_across_heads = bool(share_across_heads)
        self.prototype_logits = nn.Parameter(torch.randn(self.codebook_size, self.num_bins))
        nn.init.normal_(self.prototype_logits, std=0.02)

    def forward(self, attn_probs: torch.Tensor):
        if attn_probs.dim() != 4:
            raise ValueError(f"Expected attention probs [B, H, Q, K], got {tuple(attn_probs.shape)}")
        if attn_probs.shape[-1] != self.num_bins:
            raise ValueError(
                f"Attention quantizer expected K={self.num_bins}, got {attn_probs.shape[-1]}"
            )

        bsz, heads, query_len, _ = attn_probs.shape
        prototypes = self.prototype_logits.softmax(dim=-1)  # (K_codebook, K_memory)

        if self.share_across_heads:
            base = attn_probs.mean(dim=1)  # (B, Q, K_memory)
            flat = base.reshape(-1, self.num_bins)
            dist = torch.cdist(flat, prototypes, p=2)
            indices = dist.argmin(dim=-1)
            quantized = F.embedding(indices, prototypes).view(bsz, query_len, self.num_bins)
            quantized = quantized + (base - base.detach())  # straight-through
            quantized = quantized.unsqueeze(1).expand(-1, heads, -1, -1)
            indices = indices.view(bsz, query_len)
        else:
            flat = attn_probs.reshape(-1, self.num_bins)
            dist = torch.cdist(flat, prototypes, p=2)
            indices = dist.argmin(dim=-1)
            quantized = F.embedding(indices, prototypes).view(bsz, heads, query_len, self.num_bins)
            quantized = quantized + (attn_probs - attn_probs.detach())  # straight-through
            indices = indices.view(bsz, heads, query_len)

        return quantized, {
            "indices": indices,
            "codebook_size": self.codebook_size,
            "share_across_heads": self.share_across_heads,
        }


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = (
            nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        )

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            use_flash=use_flash_attn,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)  # (B, T_ctx, D_ctx) -> (B, T_ctx, D)

        # Fixed learned query tokens. Output length is always num_latents.
        latents = repeat(self.latents, "n d -> b n d", b=batch)  # (B, N_latent, D)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents  # (B, N_latent, D)
            latents = ff(latents) + latents  # (B, N_latent, D)

        return self.norm(latents)  # (B, N_latent, D)


class SpeakerTokenMixer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_head=64,
        heads=8,
        ff_mult=4,
        dropout=0.0,
        use_flash_attn=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=dropout,
                            use_flash=use_flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        hidden = x
        for attn, ff in self.layers:
            hidden = attn(hidden, mask=mask) + hidden
            hidden = ff(hidden) + hidden
        return self.norm(hidden)


class MemoryCrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
        discretize_attn=False,
        attn_codebook_size=128,
        attn_share_across_heads=True,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = (
            nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        )

        self.memory_tokens = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.memory_tokens, std=0.02)
        self.discretize_attn = bool(discretize_attn)
        self.latest_attn_quantizer_info = []

        self.layers = nn.ModuleList([])
        self.attn_pattern_quantizers = nn.ModuleList() if self.discretize_attn else None
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            use_flash=use_flash_attn,
                            cross_attn_include_queries=False,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
            if self.attn_pattern_quantizers is not None:
                self.attn_pattern_quantizers.append(
                    AttentionPatternQuantizer(
                        codebook_size=attn_codebook_size,
                        num_bins=num_latents,
                        share_across_heads=attn_share_across_heads,
                    )
                )

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        del mask  # Memory tokens are fixed-length and do not require padding masks here.

        batch = x.shape[0]
        # Input features stay as queries, so output length follows the input time axis.
        hidden = self.proj_context(x)  # (B, T_ctx, D_ctx) -> (B, T_ctx, D)
        memory = repeat(self.memory_tokens, "n d -> b n d", b=batch)  # (B, N_memory, D)
        self.latest_attn_quantizer_info = []

        for idx, (attn, ff) in enumerate(self.layers):
            if self.attn_pattern_quantizers is not None:
                hidden_attn, _, aux = attn(
                    hidden,
                    memory,
                    return_attn=True,
                    attn_postprocess=self.attn_pattern_quantizers[idx],
                )
                self.latest_attn_quantizer_info.append(aux)
            else:
                hidden_attn = attn(hidden, memory)
            hidden = hidden_attn + hidden  # (B, T_ctx, D)
            hidden = ff(hidden) + hidden  # (B, T_ctx, D)

        return self.norm(hidden)  # (B, T_ctx, D)


if __name__ == "__main__":
    model = PerceiverResampler(dim=256, dim_context=80)
    x = torch.randn(8, 200, 80)
    out = model(x)
    print(out.shape)  # [8, 32, 80]

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))
