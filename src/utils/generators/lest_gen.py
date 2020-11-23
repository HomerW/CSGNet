import torch
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda")

class LestGen:

    def __init__(self,
                 path,
                 batch_size=100,
                 train_size=10000):

        self.labels = torch.load(path + "labels.pt")
        self.labels = torch.from_numpy(self.labels)
        self.labels = self.labels.long()

        self.images = torch.load(path + "images.pt")
        self.images = torch.from_numpy(self.images)
        self.images = self.images.float()

        # pad labels with a stop symbol
        self.labels = F.pad(self.labels, (0, 1), 'constant', 399)

        self.train_size = train_size
        self.batch_size = batch_size

    def get_train_data(self):
        while True:
            ids = np.arange(self.train_size)
            np.random.shuffle(ids)
            for i in range(0, self.train_size, self.batch_size):
                batch_labels = self.labels[ids[i:i+self.batch_size]]
                batch_images = self.images[ids[i:i+self.batch_size]]

                yield (batch_images, batch_labels)
