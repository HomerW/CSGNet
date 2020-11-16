import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.Models.models import Encoder
from src.Models.models import ImitateJoint
from src.utils import read_config
from src.utils.generators.wake_sleep_gen import WakeSleepGen
import os
from ws_infer import infer_programs
from ws_train_inference import train_inference
from globals import device
import time
import sys
import matplotlib.pyplot as plt
import json

"""
Get initial pretrained CSGNet inference network
"""

def get_csgnet():
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
    imitate_net = imitate_net.to(device)

    print("pre loading model")
    pretrained_dict = torch.load(
        'trained_models/mix_len_cr_percent_equal_batch_3_13_prop_100_hdsz_2048_batch_2000_optim_adam_lr_0.001_wd_0.0_enocoderdrop_0.0_drop_0.2_step_mix_mode_12.pth',
        map_location=device
    )

    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    return imitate_net


"""
Runs the wake-sleep algorithm
"""
def wake_sleep(iterations):
    imitate_net = get_csgnet()

    self_training = False
    
    inf_epochs = 0
    gen_epochs = 0

    exp_name = sys.argv[1]
    import os
    os.system(f'mkdir {exp_name}')

    res = {'train':[], 'val':[], 'test':[], 'epochs':[]}

    num_train = 10000
    num_test = 3000
    batch_size = 250

    mode = sys.argv[2]
    print(f"MODE {mode}")
    
    for i in range(iterations):
        print(f"WAKE SLEEP ITERATION {i}")

        infer_path = f"{exp_name}/inference{i}"
        os.system(f'mkdir {infer_path}')
        infer_res = infer_programs(
            imitate_net, infer_path, num_train, num_test, batch_size, self_training
        )

        for key in infer_res:
            res[key].append(infer_res[key])

        res['epochs'].append(i)
        plt.clf()
        for key in ('train', 'val', 'test'):
            plt.plot(res['epochs'], res[key], label=key)
        plt.grid()
        plt.legend()
        plt.savefig(f"{exp_name}/cd_plot.png")
        
        inf_epochs += train_inference(
            imitate_net, infer_path + "/labels", num_train, num_test, batch_size, self_training, i, mode
        )

        with open(f'{exp_name}/all_res.org', 'w') as outfile:
            json.dump(res, outfile)

        #torch.save(imitate_net.state_dict(), f"{exp_name}/imitate_{i}.pth")
        print(f"Total inference epochs: {inf_epochs}")

        
wake_sleep(200)
