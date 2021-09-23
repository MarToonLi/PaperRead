from pathlib import Path
"""
两件事：
- 查看是否PIL图片经过toTensor方法，会经过归一化。
"""

import numpy as np

import torch
import torchvision.transforms as T
from torchvision.io import read_image

torch.manual_seed(1)


def show(imgs):
    img = T.ToPILImage()(imgs.to('cpu'))

from PIL import Image
dog1 = Image.open("/data/dog.png")
print(np.mean(np.array(dog1)))
dog2 = T.ToTensor()(dog1)
print(torch.mean(dog2))