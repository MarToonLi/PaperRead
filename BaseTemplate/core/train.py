#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.utils.tensorboard import SummaryWriter
from data.data_prefetcher import DataPrefetcher
from utils.logger import setup_logger
from utils.metric import MeterBuffer, gpu_mem_usage
from utils.checkpoint import load_ckpt, save_checkpoint
from models.model_utils import get_model_info
from models.ema import ModelEMA, is_parallel


class Trainer:
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
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name, time.strftime('%Y_%m_%d_%H_%M_%S'))

        os.makedirs(self.file_name, exist_ok=True)
        self.file_id = setup_logger(
            self.file_name,
            filename="train_log.txt",
            mode="a",
        )

    @logger.catch
    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    # ==========================================================================================================

    def train_one_iter(self):
        self.model.train()

        iter_start_time = time.time()

        inps, targets = self.train_prefetcher.next()
        inps = inps.to(self.data_type).cuda()
        targets = targets.to(torch.int64).cuda()
        targets.requires_grad = False

        # inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        self.loss = outputs["loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            loss=self.loss.cpu().detach().numpy(),
        )

    def before_train(self):
        logger.info("parser中接受到的参数列表: {}".format(self.args))
        logger.info("模型训练中 执行参数列表:\n{}".format(self.exp))

        # model related init
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.train_loader = self.exp.get_data_loader(batch_size=self.args.batch_size)

        print(self.train_loader)

        logger.info("init prefetcher, this might take one minute or less...")
        self.train_prefetcher = DataPrefetcher(self.train_loader)

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )

        # 无效
        # if self.args.occupy:
        #     occupy_mem(self.local_rank)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        # 并行处理
        if self.device_info[1] > 1:
            torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0,
                                                 world_size=1)
            # torch.distributed.init_process_group(backend="nccl")
            self.model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            self.model = model

        # Tensorboard logger
        self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    def before_epoch(self):
        logger.info("---> start train epoch {}".format(self.epoch + 1))

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            self.model.eval()
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.tblogger.add_scalar("train/loss", loss_meter["loss"].avg, self.progress_in_iter + 1)
            self.meter.clear_meters()

        # random resizing
        # if (self.progress_in_iter + 1) % 10 == 0:
        #     self.input_size = self.exp.random_resize(
        #         self.train_loader, self.epoch, self.rank, self.is_distributed
        #     )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    # ==========================================================================================================

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        val_loss = 0
        eval_loader = self.exp.evalator(batch_size=self.args.batch_size)
        self.eval_prefetcher = DataPrefetcher(eval_loader)

        inps, targets = self.eval_prefetcher.next()
        count = 0
        while inps is not None:
            count += 1
            inps = inps.to(torch.float32).cuda()
            targets = targets.to(torch.int64).cuda()
            targets.requires_grad = False
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                val_outputs = self.model(inps, targets)
            # if count % self.exp.print_interval == 0:
            #     logger.info('count: {},val_loss:{}.'.format(count, val_outputs["loss"]))
            val_loss += val_outputs["loss"]
            inps, targets = self.eval_prefetcher.next()

        self.tblogger.add_scalar("val/loss", val_loss, self.progress_in_iter + 1)

        logger.info("Val loss: {}. ".format(val_loss))
        self.save_ckpt("last_epoch", val_loss > self.best_ap)
        self.best_ap = max(self.best_ap, val_loss)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        save_model = self.ema_model.ema if self.use_model_ema else self.model
        logger.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )
