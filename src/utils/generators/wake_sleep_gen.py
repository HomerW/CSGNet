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

        with torch.no_grad():        
            if round > 0 and mode in ('P_LEST', 'F_LEST'):
                base_path = f"{labels_path.split('/')[0]}/inference"
                past_labels = torch.zeros(round, self.labels.shape[0], self.labels.shape[1]).long()
                for r in range(round):                
                    r_l = torch.load(f'{base_path}{r}/labels/lest_labels.pt')
                    if isinstance(r_l, np.ndarray):
                        r_l = torch.from_numpy(r_l)
                    past_labels[r] = r_l.long()

                past_labels = past_labels.permute(1,0,2)

                if mode == 'P_LEST':
                    samp_inds = (torch.rand(self.labels.shape[0]) * (round-1)).round().long()
                elif mode == 'F_LEST':
                    samp_inds = torch.zeros(self.labels.shape[0]).long()
                    
                self.round_past_labels = past_labels[torch.arange(samp_inds.shape[0]), samp_inds].to(device)
                
                    
        self.images = torch.load(labels_path + "real_images.pt")        

        if round > 0 and mode in ('P_LEST', 'F_LEST'):
            self.labels = torch.cat((self.labels, self.round_past_labels), dim=0)
            
        # pad labels with a stop symbol, should be correct but need to confirm this
        # since infer_programs currently outputs len 13 labels
        self.labels = F.pad(self.labels, (0, 1), 'constant', 399)
        
        self.train_size = self.labels.shape[0]
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.batch_size = batch_size

        with open("terminals.txt", "r") as file:
            self.unique_draw = file.readlines()
        for index, e in enumerate(self.unique_draw):
            self.unique_draw[index] = e[0:-1]

        self.parser = ParseModelOutput(self.unique_draw, self.max_len // 2 + 1, self.max_len, canvas_shape)
        self.expressions = self.parser.labels2exps(self.labels, self.labels.shape[1])
        # Remove the stop symbol and later part of the expression
        for index, exp in enumerate(self.expressions):
            self.expressions[index] = exp.split("$")[0]
        self.correct_programs = []
        self.mode = mode
        
    def get_train_data(self):
        while True:

            ids = np.arange(self.labels.shape[0])
            print(self.labels.shape[0])
            np.random.shuffle(ids)
            
            for i in range(0, self.train_size, self.batch_size):
                stacks = []
                batch_exp = [self.expressions[index] for index in ids[i:i+self.batch_size]]
                batch_labels = self.labels[ids[i:i+self.batch_size]]

                if self.mode == 'ST_LEST':                
                    batch_images = self.images[ids[i:i+self.batch_size]]

                for index, exp in enumerate(batch_exp):
                    program = self.parser.Parser.parse(exp)
                    # Check the validity of the expressions
                    if validity(program, len(program), len(program) - 1):
                        pass
                    else:
                        stack = np.zeros((64, 64))
                        stacks.append(stack)
                        continue

                    self.parser.sim.generate_stack(program)
                    stack = self.parser.sim.stack_t
                    stack = np.stack(stack, axis=0)
                    # pad if the program was shorter than the max_len since csgnet can only train on fixed sizes
                    stack = np.pad(stack, (((self.max_len + 1) - stack.shape[0], 0), (0, 0), (0, 0), (0, 0)))
                    stack = stack[-1, 0, :, :]
                    stacks.append(stack)

                
                lest_stacks = np.stack(stacks, 0).astype(dtype=np.float32)                
                
                if self.mode == 'ST_LEST':
                    van_stacks = batch_images
                    batch_data = torch.stack((
                        torch.from_numpy(lest_stacks),
                        torch.from_numpy(van_stacks)
                    )).transpose(0,1)[
                        torch.arange(van_stacks.shape[0]),
                        torch.rand(van_stacks.shape[0]).round().long()
                    ]
                    
                else:
                    batch_data = torch.from_numpy(lest_stacks)
                    
                yield (batch_data, batch_labels)
