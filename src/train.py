import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from hidra3.hidra3 import HIDRA
from utils.dataset import Data
from utils.training import Training


class Config():
    def __init__(self):
        self.seed = 0

        self.data_path = '../data'

        self.tide_gauge_locaitons = [
            'Koper',
            # 'Ancona',
            # 'Ortona',
            # 'Ravenna',
            # 'Tremiti',
            # 'Vieste',
            # 'Neretva',
            # 'Venice',
            # 'Sobra',
            # 'Stari Grad',
            # 'Vela Luka',
        ]

        self.test_range = pd.to_datetime('2019-05-29'), pd.to_datetime('2021-01-04')

        self.lr = .00001
        self.epochs = 20

        self.lr_std = .0001
        self.epochs_std = 40

        self.batch_size = 128
        self.weight_decay = .001

        self.num_workers = 4
        self.device = torch.device('cuda:0')

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


if __name__ == '__main__':
    print('start:', datetime.now())

    config = Config()
    training = Training(
        config,
        net=HIDRA(config.tide_gauge_locaitons),
        data=Data(config)
    )
    training.train()

    print('\nfinish:', datetime.now())
