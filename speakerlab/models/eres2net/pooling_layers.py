# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" This implementation is adapted from https://github.com/wenet-e2e/wespeaker."""

import torch
import torch.nn as nn


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """
    def __init__(self, **kwargs):
        super(TAP, self).__init__()

    def forward(self, x, lengths=None):
        pooling_mean = x.mean(dim=-1)
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """
    def __init__(self, **kwargs):
        super(TSDP, self).__init__()

    def forward(self, x, lengths=None):
        # The last dimension is the temporal axis
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    def __init__(self, **kwargs):
        super(TSTP, self).__init__()

    def forward(self, x, lengths=None):
        # The last dimension is the temporal axis
        if lengths is None:
            pooling_mean = x.mean(dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
            pooling_mean = pooling_mean.flatten(start_dim=1)
            pooling_std = pooling_std.flatten(start_dim=1)
            return torch.cat((pooling_mean, pooling_std), 1)

        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=x.device)
        lengths = lengths.long().clamp(min=0)
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

        time_dim = x.shape[-1]
        lengths = torch.clamp(lengths, max=time_dim)
        mask = torch.arange(time_dim, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.dtype)

        if len(x.shape) == 4:
            mask_4d = mask[:, None, None, :]
            denom = mask_4d.sum(dim=-1).clamp(min=1.0)
            mean = (x * mask_4d).sum(dim=-1) / denom
            var = (x.pow(2) * mask_4d).sum(dim=-1) / denom - mean.pow(2)
            std = torch.sqrt(var.clamp(min=1e-10))
            mean = mean.flatten(start_dim=1)
            std = std.flatten(start_dim=1)
            return torch.cat((mean, std), 1)

        mask_3d = mask[:, None, :]
        denom = mask_3d.sum(dim=-1).clamp(min=1.0)
        mean = (x * mask_3d).sum(dim=-1) / denom
        var = (x.pow(2) * mask_3d).sum(dim=-1) / denom - mean.pow(2)
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat((mean, std), 1)


class MaskedTSTP(nn.Module):
    """
    Temporal statistics pooling with time mask.

    lengths: valid time steps for each sample (shape [B]).
    """
    def __init__(self, **kwargs):
        super(MaskedTSTP, self).__init__()

    def forward(self, x, lengths=None):
        if lengths is None:
            # Fallback to vanilla behavior
            if len(x.shape) == 4:
                pooling_mean = x.mean(dim=-1)
                pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
                pooling_mean = pooling_mean.flatten(start_dim=1)
                pooling_std = pooling_std.flatten(start_dim=1)
                return torch.cat((pooling_mean, pooling_std), 1)
            pooling_mean = x.mean(dim=-1)
            pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
            pooling_mean = pooling_mean.flatten(start_dim=1)
            pooling_std = pooling_std.flatten(start_dim=1)
            return torch.cat((pooling_mean, pooling_std), 1)

        if not torch.is_tensor(lengths):
            lengths = torch.as_tensor(lengths, device=x.device)
        lengths = lengths.long().clamp(min=0)
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

        time_dim = x.shape[-1]
        lengths = torch.clamp(lengths, max=time_dim)
        mask = torch.arange(time_dim, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.dtype)

        if len(x.shape) == 4:
            # x: (B, C, F, T)
            mask_4d = mask[:, None, None, :]
            denom = mask_4d.sum(dim=-1).clamp(min=1.0)
            mean = (x * mask_4d).sum(dim=-1) / denom
            var = (x.pow(2) * mask_4d).sum(dim=-1) / denom - mean.pow(2)
            std = torch.sqrt(var.clamp(min=1e-10))
            mean = mean.flatten(start_dim=1)
            std = std.flatten(start_dim=1)
            return torch.cat((mean, std), 1)

        # x: (B, F, T)
        mask_3d = mask[:, None, :]
        denom = mask_3d.sum(dim=-1).clamp(min=1.0)
        mean = (x * mask_3d).sum(dim=-1) / denom
        var = (x.pow(2) * mask_3d).sum(dim=-1) / denom - mean.pow(2)
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat((mean, std), 1)


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)
