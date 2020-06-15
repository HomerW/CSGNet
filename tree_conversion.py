from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from torch.autograd.variable import Variable
from src.Models.loss import losses_joint
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer
from src.utils.generators.mixed_len_generator import Parser, \
    SimulateStack

program_len = 3
data_labels_path = "data/synthetic/one_op/expressions.txt"

with open(data_labels_path) as data_file:
    expressions = data_file.readlines()
with open("terminals.txt", "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

programs = []
parser = Parser()
for e in expressions:
    programs.append(parser.parse(e))

def prog_to_tree(prog):
    stack = []
    for node in prog:
        if node["type"] == "draw":
            # primitive
            value = unique_draw.index(f"{node['value']}({','.join(node['param'])})")
            stack.append({"value": value, "left": None, "right": None})
        else:
            # operator
            obj_2 = stack.pop()
            obj_1 = stack.pop()
            value = unique_draw.index(node['value'])
            stack.append({"value": value, "left": obj_2, "right": obj_1})
    return stack[0]

prog_trees = map(prog_to_tree, programs)
