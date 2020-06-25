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

device = torch.device("cuda")
inference_train_size = 10000
inference_test_size = 1000
vocab_size = 400
generator_hidden_dim = 256
generator_latent_dim = 20
max_len = 13

"""
Trains CSGNet to convergence on samples from generator network
TODO: train to convergence and not number of epochs
"""
def train_inference(inference_net, iter):
    config = read_config.Config("config_synthetic.yml")

    generator = WakeSleepGen(f"wake_sleep_data/generator/{iter}/labels.pt",
                             batch_size=config.batch_size,
                             train_size=inference_train_size,
                             test_size=inference_test_size,
                             canvas_shape=config.canvas_shape,
                             max_len=max_len)

    train_gen = generator.get_train_data()
    test_gen = generator.get_test_data()

    encoder_net, imitate_net = inference_net

    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        lr=config.lr)

    reduce_plat = LearningRate(
        optimizer,
        init_lr=config.lr,
        lr_dacay_fact=0.2,
        patience=config.patience)

    prev_test_loss = 1e20
    prev_test_cd = 1e20
    prev_test_iou = 0

    patience = 3
    num_worse = 0

    for epoch in range(30):
        train_loss = 0
        Accuracies = []
        imitate_net.train()
        for batch_idx in range(inference_train_size //
                               (config.batch_size * config.num_traj)):
            optimizer.zero_grad()
            loss = Variable(torch.zeros(1)).to(device).data
            for _ in range(config.num_traj):
                batch_data, batch_labels = next(train_gen)
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                batch_data = batch_data[:, :, 0:1, :, :]
                one_hot_labels = prepare_input_op(batch_labels, vocab_size)
                one_hot_labels = Variable(
                    torch.from_numpy(one_hot_labels)).to(device)
                outputs = imitate_net([batch_data, one_hot_labels, max_len])

                loss_k = (losses_joint(outputs, batch_labels, time_steps=max_len + 1) / (
                    max_len + 1)) / config.num_traj
                loss_k.backward()
                loss += loss_k.data
                del loss_k

            optimizer.step()
            train_loss += loss
            print(f"batch {batch_idx} train loss: {loss.cpu().numpy()}")

        mean_train_loss = train_loss / (inference_train_size // (config.batch_size))
        print(f"epoch {epoch} mean train loss: {mean_train_loss.cpu().numpy()}")
        imitate_net.eval()
        loss = Variable(torch.zeros(1)).to(device)
        metrics = {"cos": 0, "iou": 0, "cd": 0}
        IOU = 0
        COS = 0
        CD = 0
        for batch_idx in range(inference_test_size // config.batch_size):
            with torch.no_grad():
                batch_data, batch_labels = next(test_gen)
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                one_hot_labels = prepare_input_op(batch_labels, vocab_size)
                one_hot_labels = Variable(
                    torch.from_numpy(one_hot_labels)).to(device)
                test_outputs = imitate_net([batch_data, one_hot_labels, max_len])
                loss += (losses_joint(test_outputs, batch_labels, time_steps=max_len + 1) / (
                    max_len + 1))
                test_output = imitate_net.test([batch_data, one_hot_labels, max_len])
                pred_images, correct_prog, pred_prog = generator.parser.get_final_canvas(
                    test_output, if_just_expressions=False, if_pred_images=True)
                target_images = batch_data.cpu().numpy()[-1, :, 0, :, :].astype(dtype=bool)
                iou = np.sum(np.logical_and(target_images, pred_images),
                             (1, 2)) / \
                      np.sum(np.logical_or(target_images, pred_images),
                             (1, 2))
                cos = cosine_similarity(target_images, pred_images)
                CD += np.sum(chamfer(target_images, pred_images))
                IOU += np.sum(iou)
                COS += np.sum(cos)

        metrics["iou"] = IOU / inference_test_size
        metrics["cos"] = COS / inference_test_size
        metrics["cd"] = CD / inference_test_size

        test_losses = loss.data
        test_loss = test_losses.cpu().numpy() / (inference_test_size //
                                                 (config.batch_size))

        if test_loss >= prev_test_loss:
            num_worse += 1
        # else:
        #     num_worse = 0
        if num_worse >= patience:
            break
        prev_test_loss = test_loss

        reduce_plat.reduce_on_plateu(metrics["cd"])
        print("Epoch {}/{}=>  train_loss: {}, iou: {}, cd: {}, test_mse: {}".format(epoch, config.epochs,
                                          mean_train_loss.cpu().numpy(),
                                          metrics["iou"], metrics["cd"], test_loss,))

        print(f"CORRECT PROGRAMS: {len(generator.correct_programs)}")

        del test_losses, test_outputs

"""
Trains VAE to convergence on programs from inference network
TODO: train to convergence and not number of epochs
"""
def train_generator(generator_net, iter):
    labels = torch.load(f"wake_sleep_data/inference/{iter}/labels/labels_beam_width_5.pt", map_location=device)

    # pad with a start and stop token
    labels = np.pad(labels, ((0, 0), (1, 0)), constant_values=399)
    labels = np.pad(labels, ((0, 0), (0, 1)), constant_values=399)

    batch_size = 100

    optimizer = optim.Adam(generator_net.parameters(), lr=1e-3)

    generator_net.train()

    for epoch in range(500):
        train_loss = 0
        np.random.shuffle(labels)
        for i in range(0, len(labels), batch_size):
            batch = torch.from_numpy(labels[i:i+batch_size]).long().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = generator_net(batch)
            # remove start token for decoder labels
            batch = batch[:, 1:]
            loss = generator_net.loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"generator epoch {epoch} loss: {train_loss / (len(labels) * (labels.shape[1]-1))} \
                accuracy: {(recon_batch.permute(1, 2, 0).max(dim=1)[1] == batch).float().sum()/(batch.shape[0]*batch.shape[1])}")

    sample = np.zeros((inference_train_size+inference_test_size, max_len))
    for i in range(0, inference_train_size+inference_test_size, batch_size):
        batch_sample = torch.randn(1, batch_size, generator_latent_dim).to(device)
        batch_sample = generator_net.decode(batch_sample, timesteps=labels.shape[1] - 1).cpu()
        # (batch_size, timesteps)
        # sample = torch.argmax(sample.permute(1, 0, 2), dim=2)
        batch_sample = batch_sample.permute(1, 0, 2).max(dim=2)[1]
        # remove stop token
        batch_sample = batch_sample[:, :-1]
        sample[i:i+batch_size] = batch_sample

    os.makedirs(os.path.dirname(f"wake_sleep_data/generator/{iter}/"), exist_ok=True)
    torch.save(sample, f"wake_sleep_data/generator/{iter}/labels.pt")


"""
Infer programs on cad dataset
TODO: incorporate visually guided refinement, setting the flag won't work at the moment,
      will need to change how primitives are encoded since no longer finite set of primitives
"""
def infer_programs(inference_net, iter):
    refine = False
    save_viz = False

    config = read_config.Config("config_cad.yml")

    encoder_net, imitate_net = inference_net

    # Load the terminals symbols of the grammar
    with open("terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    beam_width = 5
    config.train_size = 10000
    imitate_net.eval()
    imitate_net.epsilon = 0
    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
    pred_expressions = []
    pred_labels = np.zeros((config.train_size, max_len))
    image_path = f"wake_sleep_data/inference/{iter}/images/"
    expressions_path = f"wake_sleep_data/inference/{iter}/expressions/"
    results_path = f"wake_sleep_data/inference/{iter}/results/"
    labels_path = f"wake_sleep_data/inference/{iter}/labels/"

    tweak_expressions_path = f"wake_sleep_data/inference/{iter}/tweak/expressions/"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(expressions_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(tweak_expressions_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)

    generator = Generator()

    train_gen = generator.train_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    Rs = 0
    CDs = 0
    Target_images = []
    for batch_idx in range(config.train_size // config.batch_size):
        with torch.no_grad():
            print(f"Inferring cad batch: {batch_idx}")
            data_ = next(train_gen)
            labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).to(device)
            data = Variable(torch.from_numpy(data_)).to(device)

        all_beams, next_beams_prob, all_inputs = imitate_net.beam_search(
            [data, one_hot_labels], beam_width, max_len)

        beam_labels = beams_parser(
            all_beams, data_.shape[1], beam_width=beam_width)

        beam_labels_numpy = np.zeros(
            (config.batch_size * beam_width, max_len), dtype=np.int32)
        Target_images.append(data_[-1, :, 0, :, :])
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

        pred_expressions += expressions
        predicted_images = image_from_expressions(parser, expressions)
        target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
        target_images_new = np.repeat(
            target_images, axis=0, repeats=beam_width)

        beam_R = np.sum(np.logical_and(target_images_new, predicted_images),
                        (1, 2)) / np.sum(np.logical_or(target_images_new, predicted_images), (1, 2))

        R = np.zeros((config.batch_size, 1))
        for r in range(config.batch_size):
            R[r, 0] = max(beam_R[r * beam_width:(r + 1) * beam_width])

        Rs += np.mean(R)

        beam_CD = chamfer(target_images_new, predicted_images)

        # select best expression by chamfer distance
        best_labels = np.zeros((config.batch_size, max_len))
        for r in range(config.batch_size):
            best_labels[r] = beam_labels[r][np.argmin(beam_CD[r * beam_width:(r + 1) * beam_width])]
        pred_labels[batch_idx*config.batch_size:batch_idx*config.batch_size + config.batch_size] = best_labels

        CD = np.zeros((config.batch_size, 1))
        for r in range(config.batch_size):
            CD[r, 0] = min(beam_CD[r * beam_width:(r + 1) * beam_width])

        CDs += np.mean(CD)

        if save_viz:
            for j in range(0, config.batch_size):
                f, a = plt.subplots(1, beam_width + 1, figsize=(30, 3))
                a[0].imshow(data_[-1, j, 0, :, :], cmap="Greys_r")
                a[0].axis("off")
                a[0].set_title("target")
                for i in range(1, beam_width + 1):
                    a[i].imshow(
                        predicted_images[j * beam_width + i - 1],
                        cmap="Greys_r")
                    a[i].set_title("{}".format(i))
                    a[i].axis("off")
                plt.savefig(
                    image_path +
                    "{}.png".format(batch_idx * config.batch_size + j),
                    transparent=0)
                plt.close("all")

                save_viz = False

    print(
        "Inferring cad average chamfer distance: {}".format(
            CDs / (config.train_size // config.batch_size)),
        flush=True)

    if refine:
        Target_images = np.concatenate(Target_images, 0)
        tweaked_expressions = []
        scores = 0
        for index, value in enumerate(pred_expressions):
            prog = parser.Parser.parse(value)
            if validity(prog, len(prog), len(prog) - 1):
                optim_expression, score = optimize_expression(
                    value,
                    Target_images[index // beam_width],
                    metric="chamfer",
                    max_iter=None)
                print(value)
                tweaked_expressions.append(optim_expression)
                scores += score
            else:
                # If the predicted program is invalid
                tweaked_expressions.append(value)
                scores += 16

        print("chamfer scores", scores / len(tweaked_expressions))
        with open(
                tweak_expressions_path +
                "chamfer_tweak_expressions_beamwidth_{}.txt".format(beam_width),
                "w") as file:
            for index, value in enumerate(tweaked_expressions):
                file.write(value + "\n")

    Rs = Rs / (config.train_size // config.batch_size)
    CDs = CDs / (config.train_size // config.batch_size)
    print(Rs, CDs)
    if refine:
        results = {
            "iou": Rs,
            "chamferdistance": CDs,
            "tweaked_chamfer_distance": scores / len(tweaked_expressions)
        }
    else:
        results = {"iou": Rs, "chamferdistance": CDs}

    with open(expressions_path +
              "expressions_beamwidth_{}.txt".format(beam_width), "w") as file:
        for e in pred_expressions:
            file.write(e + "\n")

    with open(results_path + "results_beam_width_{}.org".format(beam_width),
              'w') as outfile:
        json.dump(results, outfile)

    torch.save(pred_labels, labels_path + "labels_beam_width_{}.pt".format(beam_width))

"""
Get initial pretrained CSGNet inference network
"""
def get_csgnet():
    config = read_config.Config("config_synthetic.yml")

    # Encoder
    encoder_net = Encoder(config.encoder_drop)
    encoder_net.to(device)

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
    imitate_net.to(device)

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

    return (encoder_net, imitate_net)

def load_generate(iter):
    generator = WakeSleepGen(f"wake_sleep_data/generator/{iter}/labels.pt")

    train_gen = generator.get_train_data()

    batch_data, batch_labels = next(train_gen)
    f, a = plt.subplots(1, 10, figsize=(30, 3))
    for j in range(10):
        a[j].imshow(batch_data[-1, j, 0, :, :], cmap="Greys_r")
        a[j].axis("off")
    plt.savefig("10.png")
    plt.close("all")

def load_infer(iter):
    encoder_net, imitate_net = get_csgnet()
    print("pre loading model")
    pretrained_dict = torch.load(f"trained_models/imitate-{iter}.pth")
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    pretrained_dict = torch.load(f"trained_models/encoder-{iter}.pth")
    encoder_net_dict = encoder_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in encoder_net_dict
    }
    encoder_net_dict.update(pretrained_dict)
    encoder_net.load_state_dict(encoder_net_dict)

    infer_programs((encoder_net, imitate_net), iter)

"""
Runs the wake-sleep algorithm
"""
def wake_sleep(iterations):
    encoder_net, imitate_net = get_csgnet()
    generator_net = VAE(generator_hidden_dim, generator_latent_dim, vocab_size).to(device)

    for i in range(iterations):
        print(f"WAKE SLEEP ITERATION {i}")
        if not i == 0: # already inferred initial cad programs using pretrained model
            infer_programs((encoder_net, imitate_net), i)
        train_generator(generator_net, i)
        train_inference((encoder_net, imitate_net), i)

        torch.save(imitate_net.state_dict(), f"trained_models/imitate-{i}.pth")
        torch.save(encoder_net.state_dict(), f"trained_models/encoder-{i}.pth")
        torch.save(generator_net.state_dict(), f"trained_models/generator-{i}.pth")

wake_sleep(50)
# load_generate(16)
