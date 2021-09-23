# -*-coding:utf-8-*-
"""
pytorch transform的使用:
https://pytorch.org/vision/stable/transforms.html
- 大多数的变换,既可以接受PIL图片又可以接受tensor图片,但是依旧存在一些变换操作,仅仅接受其一;为此提供了两种转换方法:
torchvision.transforms.ToPILImage(mode=None) 和 torchvision.transforms.ToTensor.
    - ToPILImage: Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
        mode: 表示数据的颜色空间和像素深度;
        - 如果是四个通道,mode被认定为RGBA;
        - 如果是三个通道,mode被认定为RGB;
        - 如果是两个通道,mode被认定为LA;
        - 如果仅仅一个通道,mode被认定为某个数据类型;
    - ToTensor: HWC的通道,同时要求元素的范围在0到255之间,的numpy.ndarray 和 PIL image,而输出的tensor对象的元素取值范围在01之间.
        需要强调的是，这里的归一化就是除以255-->img.float().div(255)
        不同于Normalize的归一化：基于默认的第二范式进行归一化。v/max(v的范式，epsilon最小值)，v表示第一维度的向量，存在n个。

- 输入的图片的shape为 BCHW;
- 期待的tensor图片的元素数值范围应该在01之间;

常用的transform:
- transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # (1) 输入并不支持 PIL image.
    # (2) Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized. 注意需要有通道,哪怕是单通道.
    # (3) 面向每一个像素.即 input - mean / std ,并不意味着将像素的范围限定在01之间.如果要取消量纲,则需要进行toTensor,或者div(255)
    # Note：不要想着通过它使数据限制在0和1之间；它主要是调节平均值和方差。
- transforms.ToTensor()  # 两个主要功能：1 使得数据对象转换为tensor对象；2 使得数据处于0和1之间。
- transforms.ToPILImage(mode=None)

"""

from torchvision import transforms


def preprocess_for_train():
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),  #
            transforms.Normalize(mean=(0.5), std=(0.5))
        ]
    )
    return train_transforms


def preprocess_for_eval():
    eval_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
        ]
    )
    return eval_transforms
