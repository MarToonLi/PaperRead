# -*-coding:utf-8-*-
"""
面向单个tensor张量对象的统计分析,属性包含:
- 全局最小值和最大值;
- 全局均值和方差;
- 张量的大小
"""
import torch


def ana_tensor(tensor):
    """

    :param tensor: torch tensor
    :return: min, max, mean, std, shape
    """
    min = torch.min(tensor)
    max = torch.max(tensor)
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    shape = tensor.size()
    return min, max, mean, std, shape
