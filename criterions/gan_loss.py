import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode != "lsgan":
            raise ValueError(f"Unsupported GAN loss mode: {mode}. Expected 'lsgan'.")

    def disc_loss(self, real: torch.Tensor, fake: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_loss = F.mse_loss(real, torch.ones_like(real))
        fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        return real_loss, fake_loss

    def gen_loss(self, fake: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake, torch.ones_like(fake))
