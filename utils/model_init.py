import torch.nn as nn
import torch

def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if isinstance(module,nn.GroupNorm):
        return
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)