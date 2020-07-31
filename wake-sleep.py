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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.refine import optimize_expression
import os
import json
import sys
from src.utils.generators.shapenet_generater import Generator
from vae import VAE
from ws_infer import infer_programs
from ws_train_inference import train_inference
from fid_score import calculate_fid_given_paths
from globals import device

inference_train_size = 10000
inference_test_size = 3000
vocab_size = 400
generator_hidden_dim = 2048
generator_latent_dim = 20
max_len = 13

"""
Trains VAE to convergence on programs from inference network
TODO: train to convergence and not number of epochs
"""
def train_generator(generator_net, iter):
    # labels = torch.load(f"wake_sleep_data/inference/{iter}/labels/labels.pt", map_location=device)
    labels = torch.load(f"wake_sleep_data/best_labels_full/labels.pt", map_location=device)

    # pad with a start and stop token
    labels = np.pad(labels, ((0, 0), (1, 0)), constant_values=399)
    labels = np.pad(labels, ((0, 0), (0, 1)), constant_values=399)

    batch_size = 100

    optimizer = optim.Adam(generator_net.parameters(), lr=1e-3)

    generator_net.train()

    for epoch in range(1000000):
        train_loss = 0
        np.random.shuffle(labels)
        for i in range(0, len(labels), batch_size):
            batch = torch.from_numpy(labels[i:i+batch_size]).long().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = generator_net(batch)
            loss = generator_net.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"generator epoch {epoch} loss: {train_loss / (len(labels) * (labels.shape[1]-1))} \
                accuracy: {(recon_batch.permute(1, 2, 0).max(dim=1)[1] == batch[:, 1:]).float().sum()/(batch.shape[0]*batch.shape[1])}")

        if (epoch + 100) % 1 == 0:
            latents = torch.randn(1, inference_test_size, generator_latent_dim).to(device)
            sample = generator_net.decode(latents, timesteps=labels.shape[1] - 1).cpu().permute(1, 0, 2).max(dim=2)[1][:, :-1].numpy()
            os.makedirs(os.path.dirname(f"wake_sleep_data/generator/tmp/"), exist_ok=True)
            os.makedirs(os.path.dirname(f"wake_sleep_data/generator/tmp/val/"), exist_ok=True)
            torch.save(sample, f"wake_sleep_data/generator/tmp/labels.pt")
            torch.save(sample, f"wake_sleep_data/generator/tmp/val/labels.pt")
            fid_value = calculate_fid_given_paths(f"wake_sleep_data/generator/tmp",
                                                  "trained_models/fid-model.pth",
                                                  100,
                                                  32)
            print('FID: ', fid_value)

    train_sample = np.zeros((inference_train_size, max_len))
    for i in range(0, inference_train_size, batch_size):
        batch_sample = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_sample = generator_net.decode(batch_sample, timesteps=labels.shape[1] - 1).cpu()
        # (batch_size, timesteps)
        # sample = torch.argmax(sample.permute(1, 0, 2), dim=2)
        batch_sample = batch_sample.permute(1, 0, 2).max(dim=2)[1]
        # remove stop token
        batch_sample = batch_sample[:, :-1]
        train_sample[i:i+batch_size] = batch_sample
    test_sample = np.zeros((inference_test_size, max_len))
    for i in range(0, inference_test_size, batch_size):
        batch_sample = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_sample = generator_net.decode(batch_sample, timesteps=labels.shape[1] - 1).cpu()
        # (batch_size, timesteps)
        # sample = torch.argmax(sample.permute(1, 0, 2), dim=2)
        batch_sample = batch_sample.permute(1, 0, 2).max(dim=2)[1]
        # remove stop token
        batch_sample = batch_sample[:, :-1]
        test_sample[i:i+batch_size] = batch_sample

    os.makedirs(os.path.dirname(f"wake_sleep_data/generator/{iter}/"), exist_ok=True)
    torch.save(train_sample, f"wake_sleep_data/generator/{iter}/labels.pt")
    os.makedirs(os.path.dirname(f"wake_sleep_data/generator/{iter}/val/"), exist_ok=True)
    torch.save(test_sample, f"wake_sleep_data/generator/{iter}/val/labels.pt")

    # if not iter == 0:
    #     fid_value = calculate_fid_given_paths(f"wake_sleep_data/generator/{iter}",
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

def load_generate(iter):
    generator = WakeSleepGen(f"wake_sleep_data_tree/generator/tmp/labels.pt",
                             f"wake_sleep_data_tree/generator/tmp/val/labels.pt",
                             train_size=3000,
                             test_size=3000,)

    train_gen = generator.get_train_data()

    batch_data, batch_labels = next(train_gen)
    print(batch_labels[:10])
    f, a = plt.subplots(1, 10, figsize=(30, 3))
    for j in range(10):
        a[j].imshow(batch_data[-1, j, 0, :, :], cmap="Greys_r")
        a[j].axis("off")
    plt.savefig("10.png")
    plt.close("all")

def load_infer(iter):
    encoder_net, imitate_net = get_csgnet()
    print("pre loading model")
    # pretrained_dict = torch.load(f"trained_models/imitate-{iter}.pth")
    pretrained_dict = torch.load(f"trained_models/best-model-full.pth")
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    infer_programs((encoder_net, imitate_net), iter)

"""
Runs the wake-sleep algorithm
"""
def wake_sleep(iterations):
    # imitate_net = get_csgnet()
    generator_net = VAE(generator_hidden_dim, generator_latent_dim, vocab_size).to(device)

    # print("pre loading model")
    # pretrained_dict = torch.load("trained_models/best-model-full.pth", map_location=device)
    # imitate_net_dict = imitate_net.state_dict()
    # imitate_pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items() if k in imitate_net_dict
    # }
    # imitate_net_dict.update(imitate_pretrained_dict)
    # imitate_net.load_state_dict(imitate_net_dict)
    #
    # pretrained_dict = torch.load("trained_models/generator-3.pth", map_location=device)
    # generator_net_dict = generator_net.state_dict()
    # generator_pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items() if k in generator_net_dict
    # }
    # generator_net_dict.update(generator_pretrained_dict)
    # generator_net.load_state_dict(generator_net_dict)

    for i in range(iterations):
        print(f"WAKE SLEEP ITERATION {i}")
        # if not i == 0: # already inferred initial cad programs using pretrained model
        #     infer_programs(imitate_net, i)
        train_generator(generator_net, i)
        # train_inference(imitate_net, i)

        # torch.save(imitate_net.state_dict(), f"trained_models/imitate-{i}.pth")
        # torch.save(generator_net.state_dict(), f"trained_models/generator-{i}.pth")

wake_sleep(1)
#load_generate(0)
