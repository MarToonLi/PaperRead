# -*-coding:utf-8-*-

import torch, uuid, random
import numpy as np
from loguru import logger


def worker_init_reset_seed(worker_id):
    """
    这里生成的seed 不需要记录;
    :param worker_id:
    :return:
    """
    seed = uuid.uuid4().int % 2 ** 32

    random.seed(seed)
    np.random.seed(seed)

    torch.set_rng_state(torch.manual_seed(seed).get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prefix = "[=DataLoader=>]"
    logger.info("{} worker_id:{} | seed:{}.".format(prefix,
                                                    worker_id,
                                                    seed))
