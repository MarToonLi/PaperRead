#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import functools
import os
import time
from collections import defaultdict, deque

import numpy as np

import torch

__all__ = [
    "AverageMeter",
    "MeterBuffer",
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
]


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    TODO:尝试之后发现效果并不好。block_mem 如果是1024，最后占用1913，使用del x后，没有改变，使用torch.cuda.empty_cache后下降为887.
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem).cuda()
    del x
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        # 这里的maxlen性质是普通的list不存在的.如果不需要该性质,则deque可以换成list.
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)  # 默认插入到右侧;
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()

    # ===================================================================================================
    # median和avg面向 self.deque变量;
    # latest\global_avg以及total面向 self.total变量;
    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None  # 这里的-1,是默认从右侧开始.

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def total(self):
        return self._total


class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            # 下一行代码是连接外接和该类所代表的字典的关口.
            # self就是一个字典.
            self[k].update(v)

    def reset(self):
        for v in self.values():
            v.reset()

    def clear_meters(self):
        for v in self.values():
            v.clear()

    # get_filtered_meter 函数名略微指代的范围比较宽泛,具体地应该是get_time_related_meter.
    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}
