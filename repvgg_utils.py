from torch import nn, Tensor
import torch
from typing import Dict


def get_fused_bn_to_conv_state_dict(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Dict[str, Tensor]:
    bn_mean, bn_var, bn_gamma, bn_beta = (
        bn.running_mean,
        bn.running_var,
        bn.weight,
        bn.bias,
    )
    bn_std = (bn_var + bn.eps).sqrt()
    conv_weight = nn.Parameter((bn_gamma / bn_std).reshape(-1, 1, 1, 1) * conv.weight)
    conv_bias = nn.Parameter(bn_beta - bn_mean * bn_gamma / bn_std)
    return {'weight': conv_weight, 'bias': conv_bias}


def get_fused_conv_state_dict_from_block(block) -> Dict[str, Tensor]:
    fused_block_conv_state_dict = get_fused_bn_to_conv_state_dict(
        block.branch5[0], block.branch5[1]
    )

    if block.branch3:
        conv_3x3_state_dict = get_fused_bn_to_conv_state_dict(
            block.branch3[0], block.branch3[1]
        )
        conv_3x3_state_dict['weight'] = torch.nn.functional.pad(
            conv_3x3_state_dict['weight'], [1, 1, 1, 1]
        )
        fused_block_conv_state_dict['weight'] += conv_3x3_state_dict['weight']
        fused_block_conv_state_dict['bias'] += conv_3x3_state_dict['bias']

    if block.branch1:
        conv_1x1_state_dict = get_fused_bn_to_conv_state_dict(
            block.branch1[0], block.branch1[1]
        )
        conv_1x1_state_dict["weight"] = torch.nn.functional.pad(
            conv_1x1_state_dict["weight"], [2, 2, 2, 2]
        )
        fused_block_conv_state_dict["weight"] += conv_1x1_state_dict["weight"]
        fused_block_conv_state_dict["bias"] += conv_1x1_state_dict["bias"]

    if block.identity:
        in_channels = block.branch5[0].in_channels
        identify_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            bias=True,
            padding=2
        ).to(block.branch5[0].weight.device)
        identify_conv.weight.zero_()
        for i in range(in_channels):
            identify_conv.weight[i, i % in_channels, 2, 2] = 1

        identify_state_dict = get_fused_bn_to_conv_state_dict(
            identify_conv, block.identity
        )
        fused_block_conv_state_dict['weight'] += identify_state_dict['weight']
        fused_block_conv_state_dict['bias'] += identify_state_dict['bias']

    fused_conv_state_dict = {
        k: nn.Parameter(v) for k, v in fused_block_conv_state_dict.items()
    }

    return fused_conv_state_dict
