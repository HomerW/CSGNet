import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.Models.models import Encoder
from src.Models.models import ImitateJoint
from src.utils import read_config
from src.utils.generators.wake_sleep_gen import WakeSleepGen
import os
from kws_infer import infer_programs
from ws_train_inference import train_inference
from globals import device
import time
import sys
import matplotlib.pyplot as plt
import json

"""
Get initial pretrained CSGNet inference network
"""

def get_csgnet(mn):
    config = read_config.Config("config_synthetic.yml")
    # Encoder
    encoder_net = Encoder(config.encoder_drop)
    encoder_net = encoder_net.to(device)

    imitate_net = ImitateJoint(
        hd_sz=config.hidden_size,
        input_size=config.input_size,
        encoder=encoder_net,
        mode=config.mode,
        num_draws=400,
        canvas_shape=config.canvas_shape)
    imitate_net.load_state_dict(torch.load(f'{mn}'))
    imitate_net = imitate_net.to(device)
    
    return imitate_net


"""
Runs the wake-sleep algorithm
"""
def wake_sleep(mn, ind, name):
    imitate_net = get_csgnet(mn)

    infer_res = infer_programs(        
        imitate_net, ind, f'{name}'
    )

    """
    for i in range(int(ind)):
        infer_res = infer_programs(
            #imitate_net, 1000+i+40, f'{i+40}_{name}'
            imitate_net, i, f'{name}_{i}'
        )
    """
def main():
    """
    for i in range(6):
        path = f'kenny_lest/inference{i}/labels/best_dict.pt'
        name = f'r{i}'
        wake_sleep(path, 100, name)		
    """
    wake_sleep(sys.argv[1], sys.argv[2], sys.argv[3])
    
if __name__ == '__main__':
    main()
