import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, ParseModelOutput
from src.Models.loss import losses_joint
from src.utils import read_config
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer, beams_parser, validity, image_from_expressions, stack_from_expressions
from globals import device
import time
from src.utils.generators.shapenet_generater import Generator

inference_train_size = 10000
inference_test_size = 3000
max_len = 13
beam_width = 5

"""
Trains CSGNet to convergence on samples from generator network
TODO: train to convergence and not number of epochs
"""
def train_inference(imitate_net, path, max_epochs=None, self_training=False, all_beams=False):
    if max_epochs is None:
        epochs = 100
    else:
        epochs = max_epochs

    config = read_config.Config("config_synthetic.yml")

    if all_beams:
        train_size = inference_train_size * beam_width
    else:
        train_size = inference_train_size

    generator = WakeSleepGen(f"{path}/",
                             batch_size=config.batch_size,
                             train_size=train_size,
                             canvas_shape=config.canvas_shape,
                             max_len=max_len,
                             self_training=self_training)

    train_gen = generator.get_train_data()

    cad_generator = Generator()
    val_gen = cad_generator.val_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)


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

    best_test_cd = 1e20

    patience = 5
    num_worse = 0

    for epoch in range(epochs):
        start = time.time()
        train_loss = 0
        imitate_net.train()
        for batch_idx in range(train_size //
                               (config.batch_size * config.num_traj)):
            optimizer.zero_grad()
            loss = 0
            # acc = 0
            for _ in range(config.num_traj):
                data, labels = next(train_gen)
                # data = data[:, :, 0:1, :, :]
                one_hot_labels = prepare_input_op(labels,
                                                  len(generator.unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = data.to(device)
                labels = labels.to(device)
                outputs = imitate_net([data, one_hot_labels, max_len])
                # acc += float((torch.argmax(outputs, dim=2).permute(1, 0) == labels).float().sum()) \
                #        / (labels.shape[0] * labels.shape[1]) / config.num_traj
                loss_k = ((losses_joint(outputs, labels, time_steps=max_len + 1) / (
                    max_len + 1)) / config.num_traj)
                loss_k.backward()
                loss += float(loss_k)
                del loss_k

            optimizer.step()
            train_loss += loss
            print(f"batch {batch_idx} train loss: {loss}")
            # print(f"acc: {acc}")

        mean_train_loss = train_loss / (train_size // (config.batch_size))
        print(f"epoch {epoch} mean train loss: {mean_train_loss}")
        imitate_net.eval()
        loss = 0
        # acc = 0
        metrics = {"cos": 0, "iou": 0, "cd": 0}
        # IOU = 0
        # COS = 0
        CD = 0
        # correct_programs = 0
        # pred_programs = 0
        for batch_idx in range(inference_test_size // config.batch_size):
            parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
            with torch.no_grad():
                labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
                data_ = next(val_gen)
                one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).cuda()
                data = torch.from_numpy(data_).cuda()
                # outputs = imitate_net([data, one_hot_labels, max_len])
                # loss_k = (losses_joint(outputs, labels, time_steps=max_len + 1) /
                #          (max_len + 1))
                # loss += float(loss_k)
                test_outputs = imitate_net.test([data[-1, :, 0, :, :], one_hot_labels, max_len])
                # acc += float((torch.argmax(torch.stack(test_outputs), dim=2).permute(1, 0) == labels[:, :-1]).float().sum()) \
                #         / (len(labels) * (max_len+1)) / (inference_test_size // config.batch_size)
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_outputs, if_just_expressions=False, if_pred_images=True)
                # correct_programs += len(correct_prog)
                # pred_programs += len(pred_prog)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
                # iou = np.sum(np.logical_and(target_images, pred_images),
                #              (1, 2)) / \
                #       np.sum(np.logical_or(target_images, pred_images),
                #              (1, 2))
                # cos = cosine_similarity(target_images, pred_images)
                CD += np.sum(chamfer(target_images, pred_images))
                # IOU += np.sum(iou)
                # COS += np.sum(cos)

        # metrics["iou"] = IOU / inference_test_size
        # metrics["cos"] = COS / inference_test_size
        metrics["cd"] = CD / inference_test_size

        test_losses = loss
        test_loss = test_losses / (inference_test_size //
                                                 (config.batch_size))

        if metrics["cd"] >= best_test_cd:
            num_worse += 1
        else:
            num_worse = 0
            best_test_cd = metrics["cd"]
            best_imitate_dict = imitate_net.state_dict()
        if num_worse >= patience:
            # load the best model and stop training
            imitate_net.load_state_dict(best_imitate_dict)
            return epoch + 1

        # reduce_plat.reduce_on_plateu(metrics["cd"])
        print(f"Epoch {epoch}/100 =>  train_loss: {mean_train_loss}, iou: {0}, cd: {metrics['cd']}, test_mse: {test_loss}, test_acc: {0}")
        # print(f"CORRECT PROGRAMS: {correct_programs}")
        # print(f"PREDICTED PROGRAMS: {pred_programs}")
        # print(f"RATIO: {correct_programs/pred_programs}")

        end = time.time()
        print(f"Inference train time {end-start}")

        del test_losses, outputs, test_outputs

    return epochs
