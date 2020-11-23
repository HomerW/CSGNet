import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from src.Models.models import ImitateJoint, ParseModelOutput
from src.Models.loss import losses_joint
from src.utils import read_config
from src.utils.generators.lest_gen import LestGen
from src.utils.train_utils import prepare_input_op, chamfer, beams_parser
from src.utils.generators.shapenet_generater import Generator

device = torch.device("cuda")

"""
Trains CSGNet to convergence on samples from generator network
TODO: train to convergence and not number of epochs
"""
def train(csgnet, path, max_epochs=None):
    if max_epochs is None:
        epochs = 1000
    else:
        epochs = max_epochs
    max_len = 13
    inference_train_size = 10000
    inference_test_size = 3000

    with open("terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    config = read_config.Config("config_lest.yml")

    generator = LestGen(f"{path}/",
                        batch_size=config.batch_size,
                        train_size=inference_train_size)

    train_gen = generator.get_train_data()

    cad_generator = Generator()
    val_gen = cad_generator.val_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    optimizer = optim.Adam(
        [para for para in csgnet.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        lr=config.lr)

    best_test_loss = 1e20
    torch.save(csgnet.state_dict(), f"{path}/best_dict.pt")

    best_test_cd = 1e20

    patience = 20
    num_worse = 0

    for epoch in range(epochs):
        train_loss = 0
        csgnet.train()
        for batch_idx in range(inference_train_size //
                               (config.batch_size * config.num_traj)):
            optimizer.zero_grad()
            loss = 0
            for _ in range(config.num_traj):
                data, labels = next(train_gen)
                one_hot_labels = prepare_input_op(labels, len(unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = data.to(device)
                labels = labels.to(device)
                outputs = csgnet([data, one_hot_labels, max_len])
                loss_k = ((losses_joint(outputs, labels, time_steps=max_len + 1) / (
                    max_len + 1)) / config.num_traj)
                loss_k.backward()
                loss += float(loss_k)
                del loss_k

            optimizer.step()
            train_loss += loss
        mean_train_loss = train_loss / (inference_train_size // (config.batch_size))

        csgnet.eval()
        CD = 0
        for batch_idx in range(inference_test_size // config.batch_size):
            parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
            with torch.no_grad():
                labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
                data_ = next(val_gen)
                one_hot_labels = prepare_input_op(labels, len(unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = torch.from_numpy(data_).to(device)
                test_outputs = csgnet.test([data[-1, :, 0, :, :], one_hot_labels, max_len])
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_outputs, if_just_expressions=False, if_pred_images=True)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
                CD += np.sum(chamfer(target_images, pred_images))

        CD = CD / inference_test_size

        if CD >= best_test_cd:
            num_worse += 1
        else:
            num_worse = 0
            best_test_cd = CD
            torch.save(csgnet.state_dict(), f"{path}/best_dict.pt")
        if num_worse >= patience:
            # load the best model and stop training
            csgnet.load_state_dict(torch.load(f"{path}/best_dict.pt"))
            return epoch + 1

        print(f"Epoch {epoch}/100 =>  train loss: {mean_train_loss}, test cd: {CD}")

    return epochs
