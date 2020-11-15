import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from src.Models.models import ParseModelOutput
from src.utils.train_utils import validity
from globals import device

class LogosGen:

    def __init__(self,
                 labels_path,
                 batch_size=50,
                 train_size=150,
                 canvas_shape=[64, 64],
                 max_len=13,
                 self_training=False):

        self.images = np.load(labels_path).astype(np.float32)
        self.train_size = train_size
        self.batch_size = batch_size

    def get_train_data(self):
        while True:
            ids = np.arange(self.train_size)
            np.random.shuffle(ids)
            for i in range(0, self.train_size+self.batch_size, self.batch_size):
                batch_images = self.images[ids[i:i+self.batch_size]]
                yield batch_images
