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

    def __init__(self,
                 labels_path,
                 test_labels_path,
                 batch_size=100,
                 train_size=10000,
                 test_size=3000,
                 canvas_shape=[64, 64],
                 max_len=13):

        self.labels = torch.load(labels_path, map_location=device)
        if isinstance(self.labels, np.ndarray):
            self.labels = torch.from_numpy(self.labels).to(device)
        self.labels = self.labels.long()

        self.test_labels = torch.load(labels_path, map_location=device)
        if isinstance(self.test_labels, np.ndarray):
            self.test_labels= torch.from_numpy(self.test_labels).to(device)
        self.test_labels = self.test_labels.long()

        # pad labels with a stop symbol, should be correct but need to confirm this
        # since infer_programs currently outputs len 13 labels
        self.labels = F.pad(self.labels, (0, 1), 'constant', 399)
        self.test_labels = F.pad(self.test_labels, (0, 1), 'constant', 399)

        self.train_size = train_size
        self.test_size = test_size
        self.max_len = max_len
        self.canvas_shape = canvas_shape
        self.batch_size = batch_size

        with open("terminals.txt", "r") as file:
            self.unique_draw = file.readlines()
        for index, e in enumerate(self.unique_draw):
            self.unique_draw[index] = e[0:-1]

        self.parser = ParseModelOutput(self.unique_draw, self.max_len // 2 + 1, self.max_len, canvas_shape)
        self.expressions = self.parser.labels2exps(self.labels, self.labels.shape[1])
        self.test_expressions = self.parser.labels2exps(self.test_labels, self.test_labels.shape[1])
        # Remove the stop symbol and later part of the expression
        for index, exp in enumerate(self.expressions):
            self.expressions[index] = exp.split("$")[0]
        for index, exp in enumerate(self.test_expressions):
            self.test_expressions[index] = exp.split("$")[0]
        self.correct_programs = []

    def get_train_data(self):
        while True:
            # # full shuffle, only effective if train/test size smaller than inferred programs
            # ids = np.arange(len(self.expressions))
            # np.random.shuffle(ids)
            # self.expressions = [self.expressions[index] for index in ids]
            # self.labels = self.labels[ids]

            self.correct_programs = []
            ids = np.arange(self.train_size)
            np.random.shuffle(ids)
            for i in range(0, self.train_size, self.batch_size):
                stacks = []
                batch_exp = [self.expressions[index] for index in ids[i:i+self.batch_size]]
                batch_labels = self.labels[ids[i:i+self.batch_size]]

                for index, exp in enumerate(batch_exp):
                    program = self.parser.Parser.parse(exp)
                    # Check the validity of the expressions
                    if validity(program, len(program), len(program) - 1):
                        self.correct_programs.append(index)
                    else:
                        stack = np.zeros(
                            (self.max_len + 1, self.max_len // 2 + 1, self.canvas_shape[0],
                             self.canvas_shape[1]))
                        stacks.append(stack)
                        continue

                    self.parser.sim.generate_stack(program)
                    stack = self.parser.sim.stack_t
                    stack = np.stack(stack, axis=0)
                    # pad if the program was shorter than the max_len since csgnet can only train on fixed sizes
                    stack = np.pad(stack, (((self.max_len + 1) - stack.shape[0], 0), (0, 0), (0, 0), (0, 0)))
                    stacks.append(stack)
                stacks = np.stack(stacks, 0).astype(dtype=np.float32)

                # data needs to be (program_len + 1, dataset_size, stack_length, canvas_height, canvas_width)
                batch_data = torch.from_numpy(stacks).permute(1, 0, 2, 3, 4)
                yield (batch_data, batch_labels)

    def get_test_data(self):
        while True:
            # # full shuffle, only effective if train/test size smaller than inferred programs
            # ids = np.arange(len(self.test_expressions))
            # np.random.shuffle(ids)
            # self.test_expressions = [self.test_expressions[index] for index in ids]
            # self.test_labels = self.test_labels[ids]

            for i in range(0, self.test_size, self.batch_size):
                stacks = []
                batch_exp = self.test_expressions[i:i+self.batch_size]
                batch_labels = self.test_labels[i:i+self.batch_size]

                for index, exp in enumerate(batch_exp):
                    program = self.parser.Parser.parse(exp)
                    # Check the validity of the expressions
                    if validity(program, len(program), len(program) - 1):
                        self.correct_programs.append(index)
                    else:
                        stack = np.zeros(
                            (self.max_len + 1, self.max_len // 2 + 1, self.canvas_shape[0],
                             self.canvas_shape[1]))
                        stacks.append(stack)
                        continue

                    self.parser.sim.generate_stack(program)
                    stack = self.parser.sim.stack_t
                    stack = np.stack(stack, axis=0)
                    # pad if the program was shorter than the max_len since csgnet can only train on fixed sizes
                    stack = np.pad(stack, (((self.max_len + 1) - stack.shape[0], 0), (0, 0), (0, 0), (0, 0)))
                    stacks.append(stack)
                stacks = np.stack(stacks, 0).astype(dtype=np.float32)

                # data needs to be (program_len + 1, dataset_size, stack_length, canvas_height, canvas_width)
                batch_data = torch.from_numpy(stacks).permute(1, 0, 2, 3, 4)
                yield (batch_data, batch_labels)
