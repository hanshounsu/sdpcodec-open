import math
import torch
from torch import nn
from vq.vq.factorized_vector_quantize import FactorizedVectorQuantize

class ResidualVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        codebook_size,
        **kwargs
    ):
        super().__init__()
        print(f'Initializing ResidualVQ with {num_quantizers} quantizers and codebook sizes {codebook_size}')
        VQ = FactorizedVectorQuantize
        if type(codebook_size) == int:
            codebook_size = [codebook_size] * num_quantizers
        self.layers = nn.ModuleList([VQ(codebook_size=size, **kwargs) for size in codebook_size])
        self.num_quantizers = num_quantizers

    def forward(self, x, total_step=None, **kwargs):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []
        all_active_num = []
        
        for idx, layer in enumerate(self.layers):
            out_dict = layer(residual)
            quantized, indices, loss = out_dict["z_q"], out_dict["indices"], out_dict["vq_loss"]
            perplexity, active_num = out_dict["perplexity"], out_dict["active_num"]

            residual = residual - quantized
            
            quantized_out = quantized_out + quantized

            loss = loss.mean()

            all_indices.append(indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)
            all_active_num.append(active_num)
        all_losses, all_indices, all_perplexity, all_active_num = map(torch.stack, (all_losses, all_indices, all_perplexity, all_active_num))

        return quantized_out, all_indices, all_losses, all_perplexity, all_active_num

    def vq2emb(self, vq, proj=True):
        # [B, T, num_quantizers]
        quantized_out = 0.
        for idx, layer in enumerate(self.layers):
            quantized = layer.vq2emb(vq[:, :, idx], proj=proj)
            quantized_out = quantized_out + quantized
        return quantized_out
    def get_emb(self):
        embs = [] 
        for idx, layer in enumerate(self.layers):
            embs.append(layer.get_emb())
        return embs
