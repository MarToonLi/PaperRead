#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from copy import deepcopy

import torch
import torch.nn as nn
from thop import profile


def get_model_info(model, imgsizes: list):
    # stride = 64
    img = torch.zeros((1, 1, imgsizes[0], imgsizes[1]), device=next(model.parameters()).device)
    targets = torch.zeros((1), device=next(model.parameters()).device, dtype=torch.int64)
    FLOPs, params = profile(deepcopy(model), inputs=(img, targets,), verbose=False)
    params /= 1e6
    GFLOPs = FLOPs * 2 / pow(10, 9)
    # FLOPs /= 1e9
    # flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, GFLOPs)
    return info
