from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.amp import autocast
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        codebook_loss_weight: float = 1.0,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 0.2,
        momentum: float = 0.99,
        force_quantization_f32: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.momentum = momentum
        self.force_quantization_f32 = bool(force_quantization_f32)

        if dim != self.codebook_dim:
            self.in_project = WNConv1d(dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim, dim, kernel_size=1)

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self._codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))

    @property
    def codebook(self):
        return self._codebook

    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        # transpose since we use linear

        use_fp32_island = bool(self.force_quantization_f32 and z.is_cuda)
        autocast_ctx = (
            autocast(device_type=z.device.type, enabled=False)
            if use_fp32_island
            else nullcontext()
        )

        with autocast_ctx:
            if use_fp32_island and z.dtype not in (torch.float32, torch.float64):
                z = z.float()

            # Factorized codes project input into low-dimensional space if self.dim != self.codebook_dim
            z_e = self.in_project(z)
            z_q, indices, dists = self.decode_latents(z_e)

            # statistic the usage of codes
            embed_onehot = F.one_hot(indices, self.codebook_size).type(z_e.dtype) # [B, T, C]
            
            if (
                self.training
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                # 각 GPU의 총 샘플 수 계산
                local_count = torch.tensor(indices.numel(), device=indices.device)
                global_count = local_count.clone()
                torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
                
                # 코드 사용 빈도 합산
                code_counts = embed_onehot.sum(dim=[0, 1])  # [C]
                torch.distributed.all_reduce(code_counts, op=torch.distributed.ReduceOp.SUM)
                
                # 확률로 변환 (정확한 정규화)
                avg_probs = code_counts / global_count
                
                # code_counts를 재사용 (이미 모든 GPU의 데이터가 합산됨)
                active_counts = code_counts
            else:
                # 비분산 환경 또는 평가/테스트 모드에서는 로컬 통계 사용
                code_counts = embed_onehot.sum(dim=[0, 1])  # [C]
                avg_probs = code_counts / indices.numel()
                active_counts = code_counts

            # perplexity 계산
            eps = 1e-6 if z.dtype == torch.float16 else 1e-10
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))

            # active number 계산 (이미 동기화된 code_counts 사용)
            active_num = (active_counts > 0).sum()

            if self.training:
                # EMA 업데이트를 위한 코드 사용 통계
                code_usage = active_counts
                if code_usage.dtype == torch.float16 or code_usage.dtype == torch.bfloat16:
                    code_usage = code_usage.float()  # 임시로 float32로 변환
                    ema_inplace(self.cluster_size, code_usage, self.decay)
                else:
                    ema_inplace(self.cluster_size, code_usage, self.decay)
                # 학습 중에는 EMA로 추적한 cluster_size 기반으로 active_num 계산
                active_num = torch.sum(self.cluster_size > self.threshold_ema_dead_code).float()
            else:
                # 추론 시에는 기존 cluster_size 사용
                active_num = torch.sum(self.cluster_size > self.threshold_ema_dead_code).float()

            # if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )

            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )

            # else:
            #     commit_loss = torch.tensor(0, device=z.device)
            #     codebook_loss = torch.tensor(0, device=z.device)

            z_q = (
                z_e + (z_q - z_e).detach()
            )  # noop in forward pass, straight-through gradient estimator in backward pass

            z_q = self.out_project(z_q)

            vq_loss = (commit_loss + codebook_loss).mean()

        return {
            "z_q": z_q,
            "indices": indices,
            "dists": dists,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "active_num": active_num.float(),
        }

    def vq2emb(self, vq, out_proj=True):
        emb = self.embed_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def tokenize(self, z: torch.Tensor) -> torch.Tensor:
        """tokenize the input tensor"""
        z_e = self.in_project(z)
        _, indices, _ = self.decode_latents(z_e)
        return indices

    def detokenize(self, indices):
        """detokenize the input indices"""
        z_q = self.decode_code(indices)
        z_q = self.out_project(z_q)
        return z_q

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # with L2 normalization, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices, dist


# class FactorizedVectorQuantize(nn.Module):
#     def __init__(self, dim, codebook_size, codebook_dim, commitment, **kwargs):
#         super().__init__()
#         self.codebook_size = codebook_size
#         self.codebook_dim = codebook_dim
#         self.commitment = commitment
        
#         if dim != self.codebook_dim:
#             self.in_proj = weight_norm(nn.Linear(dim, self.codebook_dim))
#             self.out_proj = weight_norm(nn.Linear(self.codebook_dim, dim))
#         else:
#             self.in_proj = nn.Identity()
#             self.out_proj = nn.Identity()
#         self._codebook = nn.Embedding(codebook_size, self.codebook_dim)
    
#     @property
#     def codebook(self):
#         return self._codebook

#     def forward(self, z):
#         """Quantized the input tensor using a fixed codebook and returns
#         the corresponding codebook vectors

#         Parameters
#         ----------
#         z : Tensor[B x D x T]

#         Returns
#         -------
#         Tensor[B x D x T]
#             Quantized continuous representation of input
#         Tensor[1]
#             Commitment loss to train encoder to predict vectors closer to codebook
#             entries
#         Tensor[1]
#             Codebook loss to update the codebook
#         Tensor[B x T]
#             Codebook indices (quantized discrete representation of input)
#         Tensor[B x D x T]
#             Projected latents (continuous representation of input before quantization)
#         """
#         # transpose since we use linear

#         z = rearrange(z, "b d t -> b t d")

#         # Factorized codes project input into low-dimensional space
#         z_e = self.in_proj(z)  # z_e : (B x T x D)
#         z_e = rearrange(z_e, "b t d -> b d t")
#         z_q, indices = self.decode_latents(z_e)
        

#         if self.training:
#             commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction='none').mean([1, 2]) * self.commitment
#             codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction='none').mean([1, 2])
#             commit_loss = commitment_loss + codebook_loss
#         else:
#             commit_loss = torch.zeros(z.shape[0], device = z.device)

#         z_q = (
#             z_e + (z_q - z_e).detach()
#         )  # noop in forward pass, straight-through gradient estimator in backward pass

#         z_q = rearrange(z_q, "b d t -> b t d")
#         z_q = self.out_proj(z_q)
#         z_q = rearrange(z_q, "b t d -> b d t")

#         return z_q, indices, commit_loss

#     def vq2emb(self, vq, proj=True):
#         emb = self.embed_code(vq)
#         if proj:
#             emb = self.out_proj(emb)
#         return emb

#     def get_emb(self):
#         return self.codebook.weight

#     def embed_code(self, embed_id):
#         return F.embedding(embed_id, self.codebook.weight)

#     def decode_code(self, embed_id):
#         return self.embed_code(embed_id).transpose(1, 2)

#     def decode_latents(self, latents):
#         encodings = rearrange(latents, "b d t -> (b t) d")
#         codebook = self.codebook.weight  # codebook: (N x D)

#         # L2 normalize encodings and codebook
#         encodings = F.normalize(encodings)
#         codebook = F.normalize(codebook)

#         # Compute euclidean distance with codebook
#         dist = (
#             encodings.pow(2).sum(1, keepdim=True)
#             - 2 * encodings @ codebook.t()
#             + codebook.pow(2).sum(1, keepdim=True).t()
#         )
#         indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
#         z_q = self.decode_code(indices)
#         return z_q, indices
