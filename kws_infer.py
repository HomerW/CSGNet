import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from src.Models.loss import losses_joint
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer, beams_parser, validity, image_from_expressions, stack_from_expressions
import matplotlib
import matplotlib.pyplot as plt
from src.utils.refine import optimize_expression
import os
import json
from globals import device
import time
import h5py

max_len = 13
beam_width = 10

"""
Infer programs on cad dataset
"""

NUM_WRITE = 10

def infer_programs(imitate_net, ind, name):
    config = read_config.Config("config_cad.yml")
    config.batch_size = 1
    # Load the terminals symbols of the grammar
    with open("terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]
    
    imitate_net.eval()
    imitate_net.epsilon = 0
    
    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
        

    with h5py.File('data/cad/cad.h5', "r") as hf:
        images = np.array(hf.get("test_images"))

    data_ = images[int(ind)].reshape(1,1,1,64,64)
        
    labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
    one_hot_labels = prepare_input_op(labels, len(unique_draw))
    one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
    data = torch.from_numpy(data_).to(device).float()
    
    all_beams, next_beams_prob, all_inputs = imitate_net.beam_search(
        [data[-1, :, 0, :, :], one_hot_labels], beam_width, max_len)    
    
    beam_labels = beams_parser(
        all_beams, data_.shape[1], beam_width=beam_width)

    beam_labels_numpy = np.zeros(
        (config.batch_size * beam_width, max_len), dtype=np.int32)
                
    for i in range(data_.shape[1]):
        beam_labels_numpy[i * beam_width:(
            i + 1) * beam_width, :] = beam_labels[i]

    # find expression from these predicted beam labels
    expressions = [""] * config.batch_size * beam_width
    for i in range(config.batch_size * beam_width):
        for j in range(max_len):
            expressions[i] += unique_draw[beam_labels_numpy[i, j]]
    for index, prog in enumerate(expressions):
        expressions[index] = prog.split("$")[0]

    predicted_images = image_from_expressions(parser, expressions)
    target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
    target_images_new = np.repeat(
        target_images, axis=0, repeats=beam_width)

    beam_CD = chamfer(target_images_new, predicted_images)
    
    best_labels = np.zeros((config.batch_size, max_len))

    idx = np.argmin(beam_CD)
    print(expressions[idx])
    return
    #a = 1/0    
    #idx = 2
    #best_labels = beam_labels[0]#idx]
    #pred_labels = best_labels
    
    CD = np.zeros((config.batch_size, 1))
    
    #plt.imshow(predicted_images[idx], cmap="Greys_r")
    plt.imshow(data_[0, 0, 0, :, :], cmap="Greys_r")    
    plt.axis("off")
    plt.savefig(
        f"{name}_gt.png",
        #f"{ind-1000}_gt.png",
        transparent=0,
        bbox_inches = 'tight',
        pad_inches = 0
    )

    plt.imshow(predicted_images[idx], cmap="Greys_r")
    plt.axis("off")
    plt.savefig(
        f"{name}_pred.png",
        transparent=0,
        bbox_inches = 'tight',
        pad_inches = 0
    )
    return
    a = 1/0
    f, a = plt.subplots(1, beam_width + 1, figsize=(30, 3))
    a[0].imshow(data_[0, 0, 0, :, :], cmap="Greys_r")
    a[0].axis("off")
    a[0].set_title("target")
    for i in range(1, beam_width):
        a[i].imshow(
            predicted_images[i],
            cmap="Greys_r")
        a[i].set_title("{}".format(i))
        a[i].axis("off")
    plt.savefig(
        f"test.png",
        transparent=0)
    plt.close("all")
    a = 1/0
    
