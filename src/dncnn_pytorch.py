"""
KAIR-compatible DnCNN (Zhang et al., 2017) for loading official pretrained weights.

Matches cszn/KAIR: models/network_dncnn.py + models/basicblock.py (sequential, conv).

The network predicts noise n; forward() returns x - n (denoised image in [0,1] when x is [0,1]).
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def sequential(*args):
    """Flatten nested nn.Sequential (same idea as KAIR BasicSR sequential)."""
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
):
    """
    One conv stack built from a mode string, e.g. 'CBR' = Conv + BatchNorm + ReLU.
    DnCNN only needs C, B, R (and L for leaky) — see KAIR basicblock.conv.
    """
    layers = []
    for t in mode:
        if t == "C":
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "B":
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-4, affine=True))
        elif t == "R":
            layers.append(nn.ReLU(inplace=True))
        elif t == "r":
            layers.append(nn.ReLU(inplace=False))
        elif t == "L":
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        else:
            raise NotImplementedError(f"Unsupported mode character {t!r} in {mode!r}")
    return sequential(*layers)


class DnCNN(nn.Module):
    """
    DnCNN with residual learning: output = x - F(x).

    For official `dncnn_color_blind.pth` (KAIR): in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R'.
    (act_mode='R' = merged BN into convs in the released checkpoint; nb=20 not 17 for this file.)
    """

    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode="BR"):
        super().__init__()
        assert "R" in act_mode or "L" in act_mode, "act_mode must include R or L (e.g. BR, R)"
        bias = True

        # Head: first conv (+ activation from last char of act_mode, e.g. BR → 'R' → Conv+ReLU).
        m_head = conv(in_nc, nc, mode="C" + act_mode[-1], bias=bias)
        # Middle: (nb - 2) repeated conv blocks (e.g. CBR for training, CR when act_mode='R').
        m_body = [conv(nc, nc, mode="C" + act_mode, bias=bias) for _ in range(nb - 2)]
        # Tail: last conv (no ReLU) — maps back to 3 channels for RGB.
        m_tail = conv(nc, out_nc, mode="C", bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        # self.model(x) predicts noise n; subtract to recover denoised image.
        n = self.model(x)
        return x - n
