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

class WakeSleepGen:

    def __init__(
        self,
        labels_path,
        batch_size,
        train_size,
        canvas_shape,
        max_len,
        self_training,
        round,
        mode
    ):

        self.labels = torch.load(labels_path + "lest_labels.pt", map_location=device)
        if isinstance(self.labels, np.ndarray):
            self.labels = torch.from_numpy(self.labels).to(device)
        self.labels = self.labels.long()

        self.labels = F.pad(self.labels, (0, 1), 'constant', 399)
        
        self.lest_images = torch.load(labels_path + "lest_images.pt")     
        self.images = torch.load(labels_path + "real_images.pt")        
        self.train_size = train_size
        self.batch_size = batch_size

        self.max_len = max_len
        self.canvas_shape = canvas_shape
        with open("terminals.txt", "r") as file:
            self.unique_draw = file.readlines()
        for index, e in enumerate(self.unique_draw):
            self.unique_draw[index] = e[0:-1]

        
    def get_train_data(self):
        while True:

            ids = np.arange(self.labels.shape[0])
            np.random.shuffle(ids)
            
            for i in range(0, self.train_size, self.batch_size):
                batch_images = self.images[ids[i:i+self.batch_size]]
                lest_batch_images = self.lest_images[ids[i:i+self.batch_size]]
                batch_labels = self.labels[ids[i:i+self.batch_size]]
                
                van_stacks = batch_images
                batch_data = torch.stack((
                    torch.from_numpy(lest_batch_images),
                    torch.from_numpy(van_stacks)
                )).transpose(0,1)[
                    torch.arange(van_stacks.shape[0]),
                    torch.rand(van_stacks.shape[0]).round().long()
                ]
                                                    
                yield (batch_data, batch_labels)
