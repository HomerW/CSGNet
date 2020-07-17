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
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer, beams_parser, validity, image_from_expressions, stack_from_expressions
import matplotlib
import matplotlib.pyplot as plt
from src.utils.refine import optimize_expression
import os
import json
import sys
from src.utils.generators.shapenet_generater import Generator
from vae_tree_reduced import VAE
from tree_conversion import label_to_tree, tree_to_label
from ws_infer import infer_programs
from ws_train_inference import train_inference
from fid_score import calculate_fid_given_paths
from globals import device

inference_train_size = 10000
inference_test_size = 3000
vocab_size = 400
generator_hidden_dim = 256
generator_latent_dim = 20
max_len = 13

"""
Trains VAE to convergence on programs from inference network
TODO: train to convergence and not number of epochs
"""
def train_generator(generator_net, iter):
    # labels = torch.load(f"wake_sleep_data/inference/{iter}/labels/labels.pt", map_location=device)
    labels = torch.load(f"wake_sleep_data/best_labels_full/labels.pt", map_location=device)
    trees = list(map(label_to_tree, labels))

    optimizer = optim.Adam(generator_net.parameters(), lr=1e-4)

    generator_net.train()

    for epoch in range(10):
        train_loss = 0
        batch_loss = 0
        acc = 0
        np.random.shuffle(trees)
        for i, t in enumerate(trees):
            optimizer.zero_grad()
            decoder_out, mu, logvar = generator_net(t)
            loss = generator_net.loss_function(decoder_out, t, mu, logvar)
            loss.backward()
            batch_loss += float(loss)
            optimizer.step()

            label = torch.argmax(tree_to_label(decoder_out), dim=1)
            acc += (label == tree_to_label(t)).float().sum() / len(label)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(trees)}")
                print(batch_loss / 100)
                print(f"acc: {acc / (i+1)}")
                train_loss += batch_loss / 100
                batch_loss = 0



            del decoder_out, mu, logvar

        print(f"generator epoch {epoch} loss: {train_loss / (len(labels) / 100)}")

        with torch.no_grad():
            test_sample = np.zeros((inference_test_size, max_len))
            total_len = 0
            for i in range(inference_test_size):
                latent = torch.randn(generator_latent_dim).to(device)
                sampled_tree = generator_net.decode(latent)
                sampled_label = torch.argmax(tree_to_label(sampled_tree), dim=1)
                total_len += len(sampled_label)
                # print(sampled_label)
                sampled_label = F.pad(sampled_label, (0, max_len-len(sampled_label)), 'constant', 399)
                test_sample[i] = sampled_label.cpu().numpy()
            print(f"AVERAGE SAMPLE LENGTH: {total_len/inference_test_size}")
            os.makedirs(os.path.dirname(f"wake_sleep_data_tree/generator/tmp/"), exist_ok=True)
            os.makedirs(os.path.dirname(f"wake_sleep_data_tree/generator/tmp/val/"), exist_ok=True)
            torch.save(test_sample, f"wake_sleep_data_tree/generator/tmp/labels.pt")
            torch.save(test_sample, f"wake_sleep_data_tree/generator/tmp/val/labels.pt")
            fid_value = calculate_fid_given_paths(f"wake_sleep_data_tree/generator/tmp",
                                                  "trained_models/fid-model.pth",
                                                  100,
                                                  32)
            print('FID: ', fid_value)

    with torch.no_grad():
        train_sample = np.zeros((inference_train_size, max_len))
        for i in range(inference_train_size):
            latent = torch.randn(generator_latent_dim).to(device)
            sampled_tree = generator_net.decode(latent)
            sampled_label = torch.argmax(tree_to_label(sampled_tree), dim=1)
            # print(sampled_label)
            sampled_label = F.pad(sampled_label, (0, max_len-len(sampled_label)), 'constant', 399)
            train_sample[i] = sampled_label.cpu().numpy()
        test_sample = np.zeros((inference_test_size, max_len))
        for i in range(inference_test_size):
            latent = torch.randn(generator_latent_dim).to(device)
            sampled_tree = generator_net.decode(latent)
            sampled_label = torch.argmax(tree_to_label(sampled_tree), dim=1)
            sampled_label = F.pad(sampled_label, (0, max_len-len(sampled_label)), 'constant', 399)
            test_sample[i] = sampled_label.cpu().numpy()

    os.makedirs(os.path.dirname(f"wake_sleep_data_tree/generator/{iter}/"), exist_ok=True)
    torch.save(train_sample, f"wake_sleep_data_tree/generator/{iter}/labels.pt")
    os.makedirs(os.path.dirname(f"wake_sleep_data_tree/generator/{iter}/val/"), exist_ok=True)
    torch.save(test_sample, f"wake_sleep_data_tree/generator/{iter}/val/labels.pt")

    # if not iter == 0:
    #     fid_value = calculate_fid_given_paths(f"wake_sleep_data_tree/generator/{iter}",
    #                                           f"trained_models/best-model-full.pth",
    #                                           100)
    #     print('FID: ', fid_value)

"""
Get initial pretrained CSGNet inference network
"""
def get_csgnet():
    config = read_config.Config("config_synthetic.yml")

    # Encoder
    encoder_net = Encoder(config.encoder_drop)
    encoder_net = encoder_net.to(device)

    # Load the terminals symbols of the grammar
    with open("terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    imitate_net = ImitateJoint(
        hd_sz=config.hidden_size,
        input_size=config.input_size,
        encoder=encoder_net,
        mode=config.mode,
        num_draws=len(unique_draw),
        canvas_shape=config.canvas_shape)
    imitate_net = imitate_net.to(device)

    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath, map_location=device)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    for param in imitate_net.parameters():
        param.requires_grad = True

    for param in encoder_net.parameters():
        param.requires_grad = True

    return imitate_net

"""
Runs the wake-sleep algorithm
"""
def wake_sleep(iterations):
    imitate_net = get_csgnet()
    generator_net = VAE(generator_hidden_dim, generator_latent_dim, vocab_size-1, max_len).to(device)

    # print("pre loading model")
    # pretrained_dict = torch.load("trained_models_tree/generator-0.pth", map_location=device)
    # generator_net_dict = generator_net.state_dict()
    # generator_net_pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items() if k in generator_net_dict
    # }
    # generator_net_dict.update(generator_net_pretrained_dict)
    # generator_net.load_state_dict(generator_net_dict)

    for i in range(iterations):
        print(f"WAKE SLEEP ITERATION {i}")
        # if not i == 0: # already inferred initial cad programs using pretrained model
        #     infer_programs(imitate_net, i)
        train_generator(generator_net, i)
        # train_inference(imitate_net, i)

        # torch.save(imitate_net.state_dict(), f"trained_models_tree/imitate-{i}.pth")
        torch.save(generator_net.state_dict(), f"trained_models_tree/generator-{i}.pth")

wake_sleep(1)
