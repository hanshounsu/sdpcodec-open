# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
from torch import nn, sin, pow
import torch.nn.functional as F
from torch.nn import Parameter


def _apply_split_condition_conv1d(
    prenet: nn.Conv1d,
    condition,
    target_length: int,
) -> torch.Tensor:
    """Apply a 1x1 condition conv without materializing repeated global condition tensors."""
    if not isinstance(condition, tuple):
        if condition.ndim == 2:
            condition = condition.unsqueeze(-1)
        if condition.shape[-1] != target_length:
            condition = torch.nn.functional.interpolate(condition, size=target_length, mode='nearest')
        return prenet(condition)

    global_condition, time_condition = condition
    global_dim = 0 if global_condition is None else int(global_condition.shape[1])
    time_dim = 0 if time_condition is None else int(time_condition.shape[1])
    expected_dim = int(prenet.in_channels)
    if global_dim + time_dim != expected_dim:
        raise ValueError(
            f"Condition split mismatch: expected {expected_dim} channels, "
            f"got global={global_dim}, time={time_dim}."
        )

    target_tensor = time_condition if time_condition is not None else global_condition
    weight = prenet.weight.to(device=target_tensor.device, dtype=target_tensor.dtype)
    bias = (
        prenet.bias.to(device=target_tensor.device, dtype=target_tensor.dtype)
        if prenet.bias is not None else None
    )
    out = None

    if time_condition is not None:
        if time_condition.ndim == 2:
            time_condition = time_condition.unsqueeze(-1)
        if time_condition.shape[-1] != target_length:
            time_condition = torch.nn.functional.interpolate(time_condition, size=target_length, mode='nearest')
        time_weight = weight[:, global_dim:, :]
        out = F.conv1d(time_condition, time_weight, bias=bias)
    elif bias is not None:
        out = bias.view(1, -1, 1).expand(-1, -1, target_length)

    if global_condition is not None:
        global_weight = weight[:, :global_dim, 0]
        global_out = F.linear(global_condition, global_weight).unsqueeze(-1)
        out = global_out if out is None else out + global_out

    if out is None:
        batch_size = 1 if global_condition is None else int(global_condition.shape[0])
        out = weight.new_zeros((batch_size, prenet.out_channels, target_length))

    return out


class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBetaWithCondition(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Condition: (B, D), where D-dimension will be mapped to C dimensions
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
        - condition_alpha_prenet - trainable parameter that controls alpha and beta using condition
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256, 128)
        >>> x = torch.randn(256)
        >>> cond = torch.randn(128)
        >>> x = a1(x, cond)
    '''
    def __init__(self, in_features, condition_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: dimension of the input
            - condition_features: dimension of the condition vectors
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha, beta will be trained along with the rest of your model.
        '''
        super(SnakeBetaWithCondition, self).__init__()
        self.in_features = in_features
        
        self.condition_alpha_prenet = torch.nn.Linear(condition_features, in_features)
        # self.condition_beta_prenet = torch.nn.Linear(condition_features, in_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x, condition):
        '''
        condition: [B, D]
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        
        condition = torch.tanh(self.condition_alpha_prenet(condition).unsqueeze(-1))  # Same prenet for both alpha and beta, to save parameters
        alpha = alpha + condition
        beta = beta + 0.5 * condition  # multiply 0.5 for avoiding beta being too small
        
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBetaWithTimeVaryingCondition(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    with time-varying condition support
    Shape:
        - Input: (B, C, T)
        - Condition: (B, D, T), where D-dimension will be mapped to C dimensions
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
        - condition_alpha_prenet - trainable parameter that controls alpha and beta using condition
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = SnakeBetaWithTimeVaryingCondition(256, 128)
        >>> x = torch.randn(8, 256, 1000)  # (B, C, T)
        >>> cond = torch.randn(8, 128, 1000)  # (B, D, T)
        >>> x = a1(x, cond)
    '''
    def __init__(self, in_features, condition_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: dimension of the input
            - condition_features: dimension of the condition vectors
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha, beta will be trained along with the rest of your model.
        '''
        super(SnakeBetaWithTimeVaryingCondition, self).__init__()
        self.in_features = in_features
        
        # 1D Conv for time-varying condition processing
        self.condition_alpha_prenet = torch.nn.Conv1d(condition_features, in_features, kernel_size=1)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x, condition):
        '''
        x: [B, C, T]
        condition: [B, D, T]
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        
        condition = torch.tanh(
            _apply_split_condition_conv1d(
                self.condition_alpha_prenet,
                condition,
                target_length=x.shape[-1],
            )
        )
        
        # Apply time-varying modulation
        alpha = alpha + condition
        beta = beta + 0.5 * condition  # multiply 0.5 for avoiding beta being too small
        
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
