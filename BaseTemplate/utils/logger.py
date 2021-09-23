#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import inspect
import os
import sys
from loguru import logger

def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    # 这一步,使得logger的输出信息能够呈现在终端.
    # logger.add(
    #     sys.stdout,
    #     format=loguru_format,
    #     level="INFO",  # WARNING INFO
    #     enqueue=True,  # 要记录的消息是否应在到达接收器之前首先通过多进程安全队列。这在通过多个进程记录到文件时很有用。
    # )
    file_id = logger.add(save_file, format=loguru_format, enqueue=True, level="INFO")
    return file_id

