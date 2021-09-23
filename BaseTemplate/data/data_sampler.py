# -*-coding:utf-8-*-
"""
参考链接：Sampler类与4种采样方式 https://zhuanlan.zhihu.com/p/100280685?utm_source=qq
- Sequential Sampler（顺序采样）
- Random Sampler（随机采样）
- Subset Random Sampler（子集随机采样）
- Weighted Random Sampler（加权随机采样）

1 基类Sampler
def __iter__(self):
    raise NotImplementedError

2 顺序采样Sequential Sampler
# __iter__()方法负责返回一个可迭代对象，这个可迭代对象是由range产生的顺序数值序列，也就是说迭代是按照顺序进行的
def __iter__(self):
    return iter(range(len(self.data_source)))

3 随机采样RandomSampler
# replacement 控制的应该为是否重复采样 如果为True，则代码中的表现是：randint；False则为randperm。
# 特别的，重复选取时，每次选择的随机数的取值范围是局限在一个批次中，[0,31] [32，63] ...... 每个批次重复的数据在各自的范围内选择。
# num_samples 参数仅当replacement为True时，才可以使用。指定该参数的值是为了避免运行时，dataset的数目会发生变化。
def __iter__(self):
    n = len(self.data_source)
    if self.replacement:
        # 生成的随机数是可能重复的
        for _ in range(self.num_samples // 32):
            yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
        yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
    # 生成的随机数是不重复的
    yield iter(torch.randperm(n).tolist())

4、子集随机采样Subset Random Sampler
# 返回的并不是随机数序列，而是通过随机数序列作为indices的索引，进而返回打乱的数据本身。
# Subset Random Sampler应该用于训练集、测试集和验证集的划分，
def __iter__(self):
    # 以元组形式返回不重复打乱后的“数据”
    return (self.indices[i] for i in torch.randperm(len(self.indices)))

5 批采样BatchSampler
# drop_last为“True”时，如果采样得到的数据个数小于batch_size则抛弃本个batch的数据。
# 对于__iter__()中的for循环，作用应该是以“生成器”的方式不断的从sampler中获取batch。
def __iter__(self):
    batch = []
    for idx in self.sampler:
        batch.append(idx)
        # 如果采样个数和batch_size相等则本次采样完成
        if len(batch) == self.batch_size:
            yield batch
            batch = []
    # for结束后在不需要剔除不足batch_size的采样个数时返回当前batch
    if len(batch) > 0 and not self.drop_last:
        yield batch
def __len__(self):
    # 在不进行剔除时，数据的长度就是采样器索引的长度
    if self.drop_last:
        return len(self.sampler) // self.batch_size
    else:
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
"""
from torch.utils.data.sampler import RandomSampler as torchRandomSampler
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from loguru import logger
import itertools
from torch.utils.data.sampler import Sampler
import torch


class RandomSampler(torchRandomSampler):
    def __repr__(self):
        prefix = "[=RandomSampler=>]"
        logger.info("{} len:{} | replacement:{}.".format(
            prefix,
            self.__len__(),
            self.replacement
        ))
        return ""


class BatchSampler(torchBatchSampler):
    def __repr__(self):
        prefix = "[=BatchSampler=>]"
        logger.info("{} batch_size:{} | drop_last:{} | len:{}.".format(
            prefix,
            self.batch_size,
            self.drop_last,
            self.__len__()
        ))
        return ""


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
            self,
            size: int,
            shuffle: bool = True,
            seed: int = 0,
            rank=0,
            world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
