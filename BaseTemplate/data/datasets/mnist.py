# -*-coding:utf-8-*-
import os, torch
from torch.utils.data.dataset import Dataset as torchDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import sys
from PIL import Image

sys.path.append("/")
from loguru import logger
from utils.logger import setup_logger
from data.data_transforms import preprocess_for_train, preprocess_for_eval


class MNIST_(torchDataset):
    def __init__(self, data_path="/home/cold/PaperReadFastly/data", is_training=False, transform=None):
        self.prefix = "[=Dataset=>] "
        self.is_training = is_training
        self.transform = transform()
        # 获取数据
        dataset = MNIST(root=data_path, train=self.is_training, download=True)
        # 获取到的是 bytes 字节数据,而字节数据不能用 mean  std 等数据统计方法.
        # float 会 进行数据转换.
        self.data, self.target = dataset.data.float(), dataset.targets.float()
        self.data = torch.unsqueeze(self.data, dim=1)

    def __getitem__(self, index):
        """
        如果数据集中的data是路径，则：
            img = Image.open(self.img_path[index]).convert('RGB')
        如果数据集中的data是torch数据，则：
            img = data
        :param index:数据在一个batch中的索引值。
        :return:
        """
        img, target = self.data[index], self.target[index]
        if self.transform != None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        """
        一些细节信息在这里进行输出.
        :return:
        """
        logger.info(
            "{} {} | data.shape:{} | target.shape:{}.".format(
                self.prefix,
                "get {}".format("trian dataset" if self.is_training else "eval dataset"),
                self.data.shape if isinstance(self.data, torch.Tensor) else "Not A torch.Tensor",
                self.target.shape))
        if isinstance(self.data, torch.Tensor):
            self.repr_img = self.data[0]

            logger.info(
                "              data: mean:{:.4} | std:{:.4} | max:{:.4} | min:{:.4}".format(
                    torch.mean(self.repr_img),
                    torch.std(self.repr_img),
                    torch.max(self.repr_img),
                    torch.min(self.repr_img)))

            if self.transform != None:
                self.transform_img = self.transform(self.repr_img)
                logger.info(
                    "              transform: mean:{:.4} | std:{:.4} | max:{:.4} | min:{:.4}".format(
                        torch.mean(self.transform_img),
                        torch.std(self.transform_img),
                        torch.max(self.transform_img),
                        torch.min(self.transform_img)))

        return ""


if __name__ == '__main__':
    output_dir, experiment_name = "outputs", "exp1"

    file_name = os.path.join(output_dir, experiment_name)
    file_id = setup_logger(
        file_name,
        filename="train_log.txt",
        mode="a",
    )

    logger.remove(file_id)
    logger.info("Mnist")

    train_dataset = MNIST_(is_training=True, transform=preprocess_for_train)

    logger.info("加载数据的配置信息.")
    print(train_dataset)

    from torch.utils.data.dataloader import DataLoader
    from data.data_sampler import RandomSampler, BatchSampler

    train_randomsampler = RandomSampler(data_source=train_dataset, replacement=False)
    print(train_randomsampler)

    train_batchsampler = BatchSampler(sampler=train_randomsampler, batch_size=16, drop_last=True)
    print(train_batchsampler)

    from data.data_utils import worker_init_reset_seed

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=train_batchsampler,
                                  num_workers=8,
                                  pin_memory=True,  # 如果cuda比较小，就False.
                                  worker_init_fn=worker_init_reset_seed
                                  )
    print(train_dataloader)

    from data.data_prefetcher import DataPrefetcher

    train_prefetcher = DataPrefetcher(train_dataloader)
    data, target = train_prefetcher.next()


