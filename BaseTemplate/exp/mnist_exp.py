#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# code, from:
# https://github.com/Megvii-BaseDetection/YOLOX/blob/c9fe0aae2db90adccc90f7e5a16f044bf110c816/yolox/exp/yolox_base.py

import os

import torch
import torch.nn as nn
from loguru import logger

from exp.base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = 1.00
        self.width = 1.00
        self.seed = 1

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.data_dir = None
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 100
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "warmcos"
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]  # 表示当前代码所处的文件的文件名,去除后缀.
        # -----------------  testing config ------------------ #
        self.test_size = (28, 28)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_model(self):
        from models.mnist import Mnist_CNN

        # 初始化参数
        def init_model(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        self.model = Mnist_CNN()
        self.model.apply(init_model)

        return self.model

    def get_data_loader(self, batch_size):
        from data.datasets.mnist import MNIST_
        from data.data_transforms import preprocess_for_train
        from torch.utils.data.dataloader import DataLoader
        from data.data_sampler import RandomSampler, BatchSampler, InfiniteSampler
        from data.data_utils import worker_init_reset_seed

        loader_kwargs = {
            "replacement": False,
            "batch_size": batch_size,
            "drop_last": True,
            "num_workers": 2,
            "pin_memory": True
        }
        logger.info("加载训练数据的配置信息.")

        train_dataset = MNIST_(is_training=True, transform=preprocess_for_train)
        print(train_dataset)

        train_sampler = InfiniteSampler(len(train_dataset), shuffle=True, seed=self.seed if self.seed else 0)
        # train_sampler = RandomSampler(data_source=train_dataset, replacement=loader_kwargs["replacement"])
        print(train_sampler)

        train_batchsampler = BatchSampler(sampler=train_sampler,
                                          batch_size=loader_kwargs["batch_size"],
                                          drop_last=loader_kwargs["drop_last"])
        print(train_batchsampler)
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_sampler=train_batchsampler,
                                      num_workers=loader_kwargs["num_workers"],
                                      pin_memory=loader_kwargs["pin_memory"],  # 如果cuda比较小，就False.
                                      worker_init_fn=worker_init_reset_seed
                                      )

        # train_dataloader = DataLoader(dataset=train_dataset,
        #                               batch_size=loader_kwargs["batch_size"],
        #                               shuffle=True,
        #                               num_workers=loader_kwargs["num_workers"],
        #                               pin_memory=loader_kwargs["pin_memory"],
        #                               drop_last=loader_kwargs["drop_last"],
        #                               )
        return train_dataloader

    def get_eval_loader(self, batch_size):
        from data.datasets.mnist import MNIST_
        from data.data_transforms import preprocess_for_eval
        from torch.utils.data.dataloader import DataLoader
        from data.data_sampler import BatchSampler
        from data.data_utils import worker_init_reset_seed

        eval_loader_kwargs = {
            "batch_size": batch_size,
            "drop_last": False,
            "num_workers": 2,
            "pin_memory": True
        }
        logger.info("加载验证数据的配置信息.")

        eval_dataset = MNIST_(is_training=False, transform=preprocess_for_eval)
        print(eval_dataset)

        eval_sampler = torch.utils.data.sampler.SequentialSampler(data_source=eval_dataset)
        print(eval_sampler)

        eval_batchsampler = BatchSampler(sampler=eval_sampler,
                                         batch_size=eval_loader_kwargs["batch_size"],
                                         drop_last=eval_loader_kwargs["drop_last"])
        print(eval_batchsampler)

        eval_loader = DataLoader(dataset=eval_dataset,
                                 batch_sampler=eval_batchsampler,
                                 num_workers=eval_loader_kwargs["num_workers"],
                                 pin_memory=eval_loader_kwargs["pin_memory"],  # 如果cuda比较小，就False.
                                 worker_init_fn=worker_init_reset_seed
                                 )
        # eval_loader = DataLoader(dataset=eval_dataset,
        #                               batch_size=eval_loader_kwargs["batch_size"],
        #                               shuffle=False,
        #                               num_workers=eval_loader_kwargs["num_workers"],
        #                               pin_memory=eval_loader_kwargs["pin_memory"],
        #                               drop_last=eval_loader_kwargs["drop_last"],
        #                               )

        return eval_loader

    def get_optimizer(self, batch_size):
        # if "optimizer" not in self.__dict__:  # 意味着该优化器如果已经实例化,那么就不需要再定义一次.

        # 调节优化器的学习率的位置:这里的代码仅面向程序启动时;而每个epoch结束后,会使用定义的策略.
        if self.warmup_epochs > 0:
            lr = self.warmup_lr  # 如果学习率为0,则
        else:
            lr = self.basic_lr_per_img * batch_size

        # pg0, pg1, pg2 = [], [], []
        # for k, v in self.model.named_modules():
        #     if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        #         pg2.append(v.bias)  # 所有层的偏置都不需要施加正则衰减.
        #     if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        #         pg0.append(v.weight)  # 批归一化层中的权重不施加正则衰减.
        #     elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        #         pg1.append(v.weight)  # 批归一化层以外的权重 施加正则衰减.

        # optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)
        # optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
        # optimizer.add_param_group({"params": pg2})
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum, nesterov=True)

        self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from optimizer.lr_scheduler import LRScheduler
        # warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters)
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
        )
        return scheduler

    def evalator(self, batch_size):
        self.eval_loader = self.get_eval_loader(batch_size=batch_size)
        return self.eval_loader


if __name__ == '__main__':
    exp = Exp()
    print(exp)
    print(os.path.split(os.path.realpath(__file__))[1].split(".")[0])
