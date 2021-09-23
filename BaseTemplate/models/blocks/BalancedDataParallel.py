from torch.nn.parallel import DataParallel
import torch
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                # Scatter 属于内置方法，比较难理解。
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()

        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):
    # DataParallel中的参数有：module, device_ids=None, output_device=None, dim=0
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz  # 定义0号GPU的样本数目是多少。
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        # forward函数因为在DataParallel中继承了其call函数的调用forward的功能，因此调用BalancedDataParallel时，会自动在init之后调用forward函数。
        # https://zhuanlan.zhihu.com/p/357021687
        if not self.device_ids:  # 如果不存在gpu时。
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            # 此逻辑，用于实现使0号GPU仅用于数据的整合。
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids

        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        print('len(inputs): ', str(len(inputs)))
        print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):

        bsz = inputs[0].size(self.dim)  # 获取batch_size。那么inputs的零维度是什么？
        num_dev = len(self.device_ids)  # 此处使用的device_ids是没有经过gpu0_bsz==0处理的变量。
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)  # 如果没有得到整除该怎么办？

        if gpu0_bsz < bsz_unit:  # 这是这个分配类的想要的结果。
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)  # chunk 意味着 组块
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            # 如果每个设备中的样本数目保持了一致，则调用父类中的scatter方法。
            # 如果0号GPU的样本数目远超过其他的，这样的安排对于同一种内存容量的显卡而言，没有实际应用价值。
            return super().scatter(inputs, kwargs, device_ids)

        print('bsz: ', bsz)
        print('num_dev: ', num_dev)
        print('gpu0_bsz: ', gpu0_bsz)
        print('bsz_unit: ', bsz_unit)
        print('chunk_sizes: ', chunk_sizes)  # 此处输出的元素数目，是真正用于输入数据处理的设备数目。
        # 这里的device_ids，和chunk_sizes的元素数目对应。inputs, kwargs两个参数没有被考虑。
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
