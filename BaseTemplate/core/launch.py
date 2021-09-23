#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
import sys

sys.path.append("/")
from loguru import logger

import torch
import torch.backends.cudnn as cudnn


def make_parser():
    """
    argparse 的学习 参考:
        - Python-argparse-命令行与参数解析 https://zhuanlan.zhihu.com/p/34395749.
        - https://blog.csdn.net/qq_43799400/article/details/119034026 python库Argparse中的可选参数设置 action=‘store_true‘ 的用法
            store_true 是指带触发action时为真，不触发则为假
    :return:
    """
    parser = argparse.ArgumentParser("Mnist train parser")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "-r", "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    """
    >>>(paper) cold@cold-OMEN-by-HP-Laptop-15-dc1xxx:~/PaperReadFastly/BaseTemplate/core$ python launch.py  --dist-backend "nl" haha "10"
    >>>2021-09-22 16:29:07.885 | INFO     | __main__:<module>:128 - ['haha', '10']
    因此:
    - "nargs=argparse.REMAINDER",收录的是未曾出现在已添加的参数的内容;以空格间隔以区分;
    - 不难解释,为什么merge函数中,对于一个列表要间隔取值,以获取key和value对;
    总结,opts面向exp类中未注册在parser.add_argument的参数.
    
    # store_true 是指带触发action时为真，不触发则为假
    """
    return parser


if __name__ == "__main__":
    from exp.mnist_exp import Exp
    from utils.tools import get_num_devices
    from core.train import Trainer
    from core.eval import Evaler
    import os

    args = make_parser().parse_args()

    # 这里的opts中的参数是exp类中定义的参数.
    # args中除了opts以外的参数,是exp
    exp = Exp()
    # logger.info(args.opts)  # 本页面内的logger的输出不会打印到文本中.
    # 更新模型exp的参数
    if args.opts != []:
        exp.merge(args.opts)
    # logger.info("模型训练 执行参数列表: \n{}".format(exp))

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # 利用args和exp中的参数
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    gpu_idx = "0 1 2 3"
    if num_gpu > 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_idx.split(" ")[:num_gpu])
        device_info = ["cuda:0", num_gpu]
    else:
        device_info = ["cpu", None]
    logger.info("此次Exp的执行环境: {}-{}".format(device_info[0], device_info[1]))

    # 如果希望启用该部分,需要在opts中设置seed属性.
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
        )

    # 开始训练.
    if args.eval != True:
        trainer = Trainer(exp, args, device_info)
        trainer.train()
    else:
        evaler = Evaler(exp, args, device_info)
        evaler.eval()
