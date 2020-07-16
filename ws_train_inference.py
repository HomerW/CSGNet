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
import os
import json
import sys
from globals import device

inference_train_size = 10000
inference_test_size = 3000
vocab_size = 400
generator_hidden_dim = 256
generator_latent_dim = 20
max_len = 13

"""
Trains CSGNet to convergence on samples from generator network
TODO: train to convergence and not number of epochs
"""
def train_inference(inference_net, path):
    config = read_config.Config("config_synthetic.yml")

    generator = WakeSleepGen(f"{path}/labels.pt",
                             f"{path}/val/labels.pt",
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

    best_test_loss = 1e20
    best_imitate_dict = imitate_net.state_dict()

    prev_test_cd = 1e20
    prev_test_iou = 0

    patience = 5
    num_worse = 0

    for epoch in range(50):
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

        if test_loss >= best_test_loss:
            num_worse += 1
        else:
            num_worse = 0
            best_test_loss = test_loss
            best_imitate_dict = imitate_net.state_dict()
        if num_worse >= patience:
            # load the best model and stop training
            imitate_net.load_state_dict(best_imitate_dict)
            break

        reduce_plat.reduce_on_plateu(metrics["cd"])
        print("Epoch {}/{}=>  train_loss: {}, iou: {}, cd: {}, test_mse: {}".format(epoch, config.epochs,
                                          mean_train_loss.cpu().numpy(),
                                          metrics["iou"], metrics["cd"], test_loss,))

        print(f"CORRECT PROGRAMS: {len(generator.correct_programs)}")

        del test_losses, test_outputs
