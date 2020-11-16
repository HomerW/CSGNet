import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.Models.models import Encoder
from src.Models.models import ImitateJoint
from src.utils import read_config
from src.utils.generators.wake_sleep_gen import WakeSleepGen
import matplotlib
import matplotlib.pyplot as plt
import os
from vae import VAE
from ws_infer import infer_programs
from ws_train_inference import train_inference
from fid_score import calculate_fid_given_paths
from globals import device
import time

inference_train_size = 10000
inference_test_size = 3000
vocab_size = 400
generator_hidden_dim = 128
generator_latent_dim = 20
max_len = 13

"""
Trains VAE to convergence on programs from inference network
TODO: train to convergence and not number of epochs
"""
def train_generator(generator_net, load_path, save_path, max_epochs=None):
    if max_epochs is None:
        epochs = 500
    else:
        epochs = max_epochs

    labels = torch.load(f"{load_path}/labels/labels.pt", map_location=device)
    # labels = torch.load("wake_sleep_data/inference/best_simple_labels/labels/labels.pt", map_location=device)

    # pad with a start and stop token
    labels = np.pad(labels, ((0, 0), (1, 1)), constant_values=399)

    batch_size = 100

    optimizer = optim.Adam(generator_net.parameters(), lr=1e-3)

    generator_net.train()

    best_train_loss = 1e20
    patience = 5
    num_worse = 0
    best_gen_dict = generator_net.state_dict()

    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        ce_loss = 0
        kl_loss = 0
        acc = 0
        np.random.shuffle(labels)
        for i in range(0, len(labels), batch_size):
            batch = torch.from_numpy(labels[i:i+batch_size]).long().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = generator_net(batch)
            ce, kld = generator_net.loss_function(recon_batch, batch, mu, logvar)
            loss = ce + 0.1*kld
            loss.backward()
            train_loss += loss.item() / (len(labels) * (labels.shape[1]-1))
            ce_loss += ce.item() / (len(labels) * (labels.shape[1]-1))
            kl_loss += kld.item() / (len(labels) * (labels.shape[1]-1))
            acc += (recon_batch.permute(1, 2, 0).max(dim=1)[1] == batch[:, 1:]).float().sum() / (len(labels) * (labels.shape[1]-1))
            optimizer.step()
        print(f"generator epoch: {epoch}, loss: {train_loss}, accuracy: {acc}, ce: {ce_loss}, kld: {kl_loss}")

        # if (epoch + 1) % 10 == 0:
        #     latents = torch.randn(1, inference_test_size, generator_latent_dim).to(device)
        #     sample_tokens = generator_net.decode(latents, timesteps=labels.shape[1] - 1)
        #     sample_tokens = sample_tokens.permute(1, 0, 2).max(dim=2)[1][:, :-1]
        #     os.makedirs(os.path.dirname(f"wake_sleep_data/generator/tmp/"), exist_ok=True)
        #     os.makedirs(os.path.dirname(f"wake_sleep_data/generator/tmp/val/"), exist_ok=True)
        #     torch.save(sample_tokens, f"wake_sleep_data/generator/tmp/labels.pt")
        #     torch.save(sample_tokens, f"wake_sleep_data/generator/tmp/val/labels.pt")
        #     fid_value = calculate_fid_given_paths(f"wake_sleep_data/generator/tmp",
        #                                           "trained_models/fid-model-three.pth",
        #                                           100,
        #                                           32)
        #     print('FID: ', fid_value)
        #     load_images()

        if train_loss >= best_train_loss:
            num_worse += 1
        else:
            num_worse = 0
            best_train_loss = train_loss
            best_gen_dict = generator_net.state_dict()
        if num_worse >= patience:
            # load the best model and stop training
            generator_net.load_state_dict(best_gen_dict)
            break

        end = time.time()
        print(f'gen epoch time {end-start}')

    train_tokens = torch.zeros((inference_train_size, max_len))
    for i in range(0, inference_train_size, batch_size):
        batch_latents = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_tokens = generator_net.decode(batch_latents, timesteps=labels.shape[1] - 1)
        batch_tokens = batch_tokens.permute(1, 0, 2).max(dim=2)[1][:, :-1]
        train_tokens[i:i+batch_size] = batch_tokens
    test_tokens = torch.zeros((inference_test_size, max_len))
    for i in range(0, inference_test_size, batch_size):
        batch_latents = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_tokens = generator_net.decode(batch_latents, timesteps=labels.shape[1] - 1)
        batch_tokens = batch_tokens.permute(1, 0, 2).max(dim=2)[1][:, :-1]
        test_tokens[i:i+batch_size] = batch_tokens

    os.makedirs(os.path.dirname(f"{save_path}/"), exist_ok=True)
    torch.save(train_tokens, f"{save_path}/labels.pt")
    os.makedirs(os.path.dirname(f"{save_path}/val/"), exist_ok=True)
    torch.save(test_tokens, f"{save_path}/val/labels.pt")

    fid_value = calculate_fid_given_paths(f"{save_path}",
                                          f"trained_models/fid-model-two.pth",
                                          100)
    print('FID: ', fid_value)

    return epoch + 1

def get_blank_csgnet():
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

    return imitate_net

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
    pretrained_dict = torch.load(config.pretrain_modelpath, map_location=device)
    # pretrained_dict = torch.load("trained_models/imitate2_10.pth", map_location=device)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    return imitate_net

def load_images():
    generator = WakeSleepGen(f"wake_sleep_data/generator/tmp/labels.pt",
                             f"wake_sleep_data/generator/tmp/val/labels.pt",
                             train_size=3000,
                             test_size=3000,)

    train_gen = generator.get_train_data()

    batch_data, batch_labels = next(train_gen)
    # for i in range(len(batch_labels)):
    #     print([int(x) for x in batch_labels[i]].index(399))
    f, a = plt.subplots(1, 10, figsize=(30, 3))
    for j in range(10):
        a[j].imshow(batch_data[-1, j, 0, :, :], cmap="Greys_r")
        a[j].axis("off")
    plt.savefig("10.png")
    plt.close("all")

def load_infer():
    imitate_net = get_csgnet()
    print("pre loading model")
    # pretrained_dict = torch.load(f"trained_models/imitate-{iter}.pth")
    pretrained_dict = torch.load(f"trained_models/best-simple-model.pth")
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    infer_programs(imitate_net, "wake_sleep_data/inference/best_simple_labels")

"""
Runs the wake-sleep algorithm
"""
def wake_sleep(iterations):
    imitate_net = get_csgnet()
    # generator_net = VAE().to(device)

    inf_epochs = 0
    gen_epochs = 0

    for i in range(iterations):
        print(f"WAKE SLEEP ITERATION {i}")

        if i == 0:
            infer_path = f"wake_sleep_data_frozen_lest_to_st/inference/0"
            # generate_path = f"wake_sleep_data/generator/0"
        else:
            infer_path = "wake_sleep_data_frozen_lest_to_st/inference"
            # generate_path = "wake_sleep_data/generator"
        infer_programs(imitate_net, infer_path, self_training=True, ab=None)

        # imitate_net = get_blank_csgnet()

        # gen_epochs += train_generator(generator_net, infer_path, generate_path, 1)
        inf_epochs += train_inference(imitate_net, infer_path + "/labels", self_training=True, ab=None)

        torch.save(imitate_net.state_dict(), f"trained_models/imitate_frozen_lest_to_st_{i}.pth")
        # torch.save(generator_net.state_dict(), f"trained_models/generator.pth")

        print(f"Total inference epochs: {inf_epochs}")
        # print(f"Total generator epochs: {gen_epochs}")

        # allowed_time -= infer_time + (inf_factor * inf_epochs) + (gen_factor * gen_epochs)
        # if allowed_time <= 0:
        #     break

# load_infer()
wake_sleep(200)
