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
def infer_programs(imitate_net, self_training=False, ab=None):
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

    generator = Generator()
    test_gen = generator.test_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    pred_expressions = []
    Rs = 0
    CDs = 0
    Target_images = []
    for batch_idx in range(config.test_size // config.batch_size):
        with torch.no_grad():
            print(f"Inferring test cad batch: {batch_idx}")
            data_ = next(test_gen)
            labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
            one_hot_labels = prepare_input_op(labels, len(unique_draw))
            one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
            data = torch.from_numpy(data_).to(device)

            all_beams, next_beams_prob, all_inputs = imitate_net.beam_search(
                [data[-1, :, 0, :, :], one_hot_labels], beam_width, max_len)

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

            # for j in range(0, config.batch_size):
            #     f, a = plt.subplots(1, beam_width + 1, figsize=(30, 3))
            #     a[0].imshow(data_[-1, j, 0, :, :], cmap="Greys_r")
            #     a[0].axis("off")
            #     a[0].set_title("target")
            #     for i in range(1, beam_width + 1):
            #         a[i].imshow(
            #             predicted_images[j * beam_width + i - 1],
            #             cmap="Greys_r")
            #         a[i].set_title("{}".format(i))
            #         a[i].axis("off")
            #     plt.savefig(
            #         "best_st/" +
            #         "{}.png".format(batch_idx * config.batch_size + j),
            #         transparent=0)
            #     plt.close("all")
            with open("best_lest_expressions.txt", "w") as file:
                for e in pred_expressions:
                    file.write(f"{e}\n")

    return CDs / (config.test_size // config.batch_size)

config = read_config.Config("config_synthetic.yml")
device = torch.device("cuda")
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

try:
    pretrained_dict = torch.load(f"trained_models/imitate2_30.pth", map_location=device)
except Exception as e:
    print(e)
imitate_net_dict = imitate_net.state_dict()
pretrained_dict = {
    k: v
    for k, v in pretrained_dict.items() if k in imitate_net_dict
}
imitate_net_dict.update(pretrained_dict)
imitate_net.load_state_dict(imitate_net_dict)

infer_programs(imitate_net)

# cd_list = []
# for i in range(100):
#     try:
#         pretrained_dict = torch.load(f"trained_models/imitate_ab_{i}.pth", map_location=device)
#     except Exception as e:
#         print(e)
#         break
#     imitate_net_dict = imitate_net.state_dict()
#     pretrained_dict = {
#         k: v
#         for k, v in pretrained_dict.items() if k in imitate_net_dict
#     }
#     imitate_net_dict.update(pretrained_dict)
#     imitate_net.load_state_dict(imitate_net_dict)
#
#     cd = infer_programs(imitate_net)
#     print(f"TEST CD: {cd}")
#     cd_list.append(cd)
#
#     with open("ws_ab_test.txt", "w") as file:
#         for c in cd_list:
#             file.write(f"{c}\n")
