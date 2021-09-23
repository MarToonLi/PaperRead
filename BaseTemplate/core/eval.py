#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from data.data_prefetcher import DataPrefetcher
from utils.logger import setup_logger
from utils.metric import MeterBuffer, gpu_mem_usage
from models.model_utils import get_model_info


class Evaler:
    def __init__(self, exp, args, device_info):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        self.device_info = device_info

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.device = device_info[0]

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join("eval", args.experiment_name, time.strftime('%Y_%m_%d_%H_%M_%S'))

        os.makedirs(self.file_name, exist_ok=True)
        self.file_id = setup_logger(
            self.file_name,
            filename="eval_log.txt",
            mode="a",
        )

    @logger.catch
    def eval(self):
        self.before_eval()
        try:
            self.eval_in_epoch()
        except Exception:
            raise
        finally:
            self.after_eval()

    def eval_in_epoch(self):
        for self.iter in range(self.max_iter_eval):
            self.eval_one_iter()

    # ==========================================================================================================

    def eval_one_iter(self):
        self.model.train()
        iter_start_time = time.time()
        inps, targets = self.eval_prefetcher.next()
        inps = inps.to(self.data_type).cuda()
        targets = targets.to(torch.int64).cuda()
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        self.loss = outputs["loss"]

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            loss=self.loss.cpu().detach().numpy(),
        )
        logger.info("count:{}, loss:{}".format(self.iter, self.loss))

    def before_eval(self):
        logger.info("parser中接受到的参数列表: {}".format(self.args))
        logger.info("模型训练中 执行参数列表:\n{}".format(self.exp))

        eval_loader = self.exp.evalator(batch_size=self.args.batch_size)
        self.eval_prefetcher = DataPrefetcher(eval_loader)
        self.max_iter_eval = len(eval_loader)

        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        model = self.resume_train(model)

        if self.device_info[1] > 1:
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0,
                                                 world_size=1)
            # torch.distributed.init_process_group(backend="nccl")
            self.model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            self.model = model

        logger.info("Evaluating start...")
        logger.info("\n{}".format(model))

    def after_eval(self):

        left_iters = self.max_iter_eval * 1 - (self.progress_in_iter + 1)
        eta_seconds = self.meter["iter_time"].global_avg * left_iters
        eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

        progress_str = " iter: {}/{}".format(
            self.iter + 1, self.max_iter_eval
        )
        loss_meter = self.meter.get_filtered_meter("loss")
        loss_str = ", ".join(
            ["{}: {:.4f}".format(k, v.latest) for k, v in loss_meter.items()]
        )

        time_meter = self.meter.get_filtered_meter("time")
        time_str = ", ".join(
            ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
        )

        logger.info(
            "{}, mem: {:.0f}Mb, {}, {}.".format(
                progress_str,
                gpu_mem_usage(),
                time_str,
                loss_str,
            )
            + (", size: {:d}, {}".format(self.input_size[0], eta_str))
        )
        self.meter.clear_meters()

    @property
    def progress_in_iter(self):
        return 1 * self.max_iter_eval + self.iter

    # ==========================================================================================================

    def resume_train(self, model):
        logger.info("resume training")
        if self.args.ckpt is None:
            ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
        else:
            ckpt_file = self.args.ckpt

        ckpt = torch.load(ckpt_file, map_location=self.device)
        # resume the model/optimizer state dict
        model.load_state_dict(ckpt["model"])
        # resume the training states variables
        logger.info(
            "loaded checkpoint '{}' ".format(
                self.args.resume
            )
        )  # noqa

        return model
