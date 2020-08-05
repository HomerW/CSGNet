import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.Models.models_perturb import Encoder
from src.Models.models_perturb import ImitateJoint
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
def train_generator(generator_net, load_path, save_path):
    # labels, perturbs = torch.load(f"{load_path}/labels/labels.pt", map_location=device)
    labels = torch.load(f"wake_sleep_data/best_labels_full/labels.pt", map_location=device)
    perturbs = np.zeros((len(labels), max_len, 3))

    # pad with a start and stop token
    labels = np.pad(labels, ((0, 0), (1, 1)), constant_values=399)
    perturbs = np.pad(perturbs, ((0, 0), (1, 1), (0, 0)), constant_values=0)

    batch_size = 100

    optimizer = optim.Adam(generator_net.parameters(), lr=1e-3)

    generator_net.train()

    for epoch in range(100000):
        train_loss = 0
        ce_loss = 0
        kl_loss = 0
        mse_loss = 0
        acc = 0
        np.random.shuffle(labels)
        for i in range(0, len(labels), batch_size):
            batch_labels = torch.from_numpy(labels[i:i+batch_size]).long().to(device)
            batch_perturb = torch.from_numpy(perturbs[i:i+batch_size]).float().to(device)
            batch = (batch_labels, batch_perturb)
            optimizer.zero_grad()
            recon_batch, mu, logvar = generator_net(batch)
            ce, mse, kld = generator_net.loss_function(recon_batch, batch, mu, logvar)
            loss = ce + mse + 0.1*kld
            loss.backward()
            train_loss += loss.item() / (len(labels) * (labels.shape[1]-1))
            ce_loss += ce.item() / (len(labels) * (labels.shape[1]-1))
            kl_loss += kld.item() / (len(labels) * (labels.shape[1]-1))
            mse_loss += mse.item() / (len(labels) * (labels.shape[1]-1))
            acc += (recon_batch[0].permute(1, 2, 0).max(dim=1)[1] == batch[0][:, 1:]).float().sum() / (len(labels) * (labels.shape[1]-1))
            optimizer.step()
        print(f"generator epoch {epoch} loss: {train_loss} accuracy: {acc} \
                ce: {ce_loss} kld: {kl_loss} mse: {mse_loss}")

        if (epoch + 1) % 10 == 0:
            latents = torch.randn(1, inference_test_size, generator_latent_dim).to(device)
            sample_tokens, sample_perturbs = generator_net.decode(latents, timesteps=labels.shape[1] - 1)
            sample_tokens = sample_tokens.permute(1, 0, 2).max(dim=2)[1][:, :-1]
            print(sample_tokens[:10])
            sample_perturbs = sample_perturbs.permute(1, 0, 2)[:, :-1]
            os.makedirs(os.path.dirname(f"wake_sleep_data/generator/tmp/"), exist_ok=True)
            os.makedirs(os.path.dirname(f"wake_sleep_data/generator/tmp/val/"), exist_ok=True)
            torch.save((sample_tokens, sample_perturbs), f"wake_sleep_data/generator/tmp/labels.pt")
            torch.save((sample_tokens, sample_perturbs), f"wake_sleep_data/generator/tmp/val/labels.pt")
            fid_value = calculate_fid_given_paths(f"wake_sleep_data/generator/tmp",
                                                  "trained_models/fid-model-latest.pth",
                                                  100,
                                                  32)
            print('FID: ', fid_value)

    train_tokens = torch.zeros((inference_train_size, max_len))
    train_perturbs = torch.zeros((inference_train_size, max_len, 3))
    for i in range(0, inference_train_size, batch_size):
        batch_latents = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_tokens, batch_perturbs = generator_net.decode(batch_latents, timesteps=labels.shape[1] - 1)
        batch_tokens = batch_tokens.permute(1, 0, 2).max(dim=2)[1][:, :-1]
        batch_perturbs = batch_perturbs.permute(1, 0, 2)[:, :-1]
        train_tokens[i:i+batch_size] = batch_tokens
        train_perturbs[i:i+batch_size] = batch_perturbs
    test_tokens = torch.zeros((inference_test_size, max_len))
    test_perturbs = torch.zeros((inference_test_size, max_len, 3))
    for i in range(0, inference_test_size, batch_size):
        batch_latents = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_tokens, batch_perturbs = generator_net.decode(batch_latents, timesteps=labels.shape[1] - 1)
        batch_tokens = batch_tokens.permute(1, 0, 2).max(dim=2)[1][:, :-1]
        batch_perturbs = batch_perturbs.permute(1, 0, 2)[:, :-1]
        test_tokens[i:i+batch_size] = batch_tokens
        test_perturbs[i:i+batch_size] = batch_perturbs

    os.makedirs(os.path.dirname(f"{save_path}/"), exist_ok=True)
    torch.save((train_tokens, train_perturbs), f"{save_path}/labels.pt")
    os.makedirs(os.path.dirname(f"{save_path}/val/"), exist_ok=True)
    torch.save((test_tokens, test_perturbs), f"{save_path}/val/labels.pt")

    fid_value = calculate_fid_given_paths(f"{save_path}",
                                          f"trained_models/fid-model.pth",
                                          100)
    print('FID: ', fid_value)

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
        canvas_shape=config.canvas_shape,
        teacher_force=True)
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

    return imitate_net

def load_images():
    generator = WakeSleepGen(f"0/labels/labels.pt",
                             f"0/labels/val/labels.pt",
                             train_size=3000,
                             test_size=3000,)

    train_gen = generator.get_train_data()

    batch_data, batch_labels, batch_perturbs = next(train_gen)
    print(batch_perturbs[:10])
    # print(any((torch.abs(batch_perturbs) > 0.5).view(-1)))
    # for i in range(len(batch_labels)):
    #     print([int(x) for x in batch_labels[i]].index(399))
    f, a = plt.subplots(1, 10, figsize=(30, 3))
    for j in range(10):
        a[j].imshow(batch_data[-1, j, 0, :, :], cmap="Greys_r")
        a[j].axis("off")
    plt.savefig("10.png")
    plt.close("all")

def load_infer(iter):
    imitate_net = get_csgnet()
    print("pre loading model")
    # pretrained_dict = torch.load(f"trained_models/imitate-{iter}.pth")
    pretrained_dict = torch.load(f"trained_models/small_test_perturb.pth")
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    infer_programs(imitate_net, iter)

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
        infer_path = f"wake_sleep_data/inference/{i}"
        generate_path = f"wake_sleep_data/generator/{i}"
        if not i == 0: # already inferred initial cad programs using pretrained model
            infer_programs(imitate_net, infer_path)
        train_generator(generator_net, infer_path, generate_path)
        # train_inference(imitate_net, infer_path + "/labels")

        # torch.save(imitate_net.state_dict(), f"trained_models/imitate.pth")
        # torch.save(generator_net.state_dict(), f"trained_models/generator.pth")

# wake_sleep(1)
load_images()
load_infer(1)
