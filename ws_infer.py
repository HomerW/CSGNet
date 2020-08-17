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
from src.utils.generators.shapenet_generater import Generator
from globals import device
import time

inference_train_size = 10000
inference_test_size = 3000
vocab_size = 400
max_len = 13
beam_width = 10

"""
Infer programs on cad dataset
"""
def infer_programs(imitate_net, path):
    save_viz = True

    config = read_config.Config("config_cad.yml")

    # Load the terminals symbols of the grammar
    with open("terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    config.train_size = 10000
    config.test_size = 3000
    imitate_net.eval()
    imitate_net.epsilon = 0
    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
    pred_expressions = []
    pred_labels = np.zeros((config.train_size, max_len))
    image_path = f"{path}/images/"
    results_path = f"{path}/results/"
    labels_path = f"{path}/labels/"

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    os.makedirs(os.path.dirname(labels_path+"val/"), exist_ok=True)

    generator = Generator()

    train_gen = generator.train_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    val_gen = generator.val_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    Rs = 0
    CDs = 0
    Target_images = []

    start = time.time()

    for batch_idx in range(config.train_size // config.batch_size):
        with torch.no_grad():
            print(f"Inferring cad batch: {batch_idx}")
            data_ = next(train_gen)
            labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
            data = torch.from_numpy(data_).to(device)

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

            # beam_R = np.sum(np.logical_and(target_images_new, predicted_images),
            #                 (1, 2)) / np.sum(np.logical_or(target_images_new, predicted_images), (1, 2))
            #
            # R = np.zeros((config.batch_size, 1))
            # for r in range(config.batch_size):
            #     R[r, 0] = max(beam_R[r * beam_width:(r + 1) * beam_width])
            #
            # Rs += np.mean(R)

            beam_CD = chamfer(target_images_new, predicted_images)

            # select best expression by chamfer distance
            best_labels = np.zeros((config.batch_size, max_len))
            for r in range(config.batch_size):
                idx = np.argmin(beam_CD[r * beam_width:(r + 1) * beam_width])
                best_labels[r] = beam_labels[r][idx]
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

    Rs = Rs / (config.train_size // config.batch_size)
    CDs = CDs / (config.train_size // config.batch_size)
    print(Rs, CDs)
    results = {"iou": Rs, "chamferdistance": CDs}

    with open(results_path + "results_beam_width_{}.org".format(beam_width),
              'w') as outfile:
        json.dump(results, outfile)

    torch.save(pred_labels, labels_path + "labels.pt")

    pred_expressions = []
    pred_labels = np.zeros((config.test_size, max_len))
    Rs = 0
    CDs = 0
    Target_images = []
    for batch_idx in range(config.test_size // config.batch_size):
        with torch.no_grad():
            print(f"Inferring val cad batch: {batch_idx}")
            data_ = next(val_gen)
            labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
            data = torch.from_numpy(data_).to(device)

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

            beam_CD = chamfer(target_images_new, predicted_images)

            CD = np.zeros((config.batch_size, 1))
            for r in range(config.batch_size):
                CD[r, 0] = min(beam_CD[r * beam_width:(r + 1) * beam_width])

            CDs += np.mean(CD)

            # select best expression by chamfer distance
            best_labels = np.zeros((config.batch_size, max_len))
            for r in range(config.batch_size):
                idx = np.argmin(beam_CD[r * beam_width:(r + 1) * beam_width])
                best_labels[r] = beam_labels[r][idx]
            pred_labels[batch_idx*config.batch_size:batch_idx*config.batch_size + config.batch_size] = best_labels

    print(
        "Inferring validation cad average chamfer distance: {}".format(
            CDs / (config.test_size // config.batch_size)),
        flush=True)

    torch.save(pred_labels, labels_path + "val/labels.pt")

    end = time.time()
    print(f"Inference time: {end-start}")
