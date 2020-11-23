import torch
import numpy as np
from src.Models.models import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.train_utils import prepare_input_op, chamfer, beams_parser, image_from_expressions
from src.utils.generators.shapenet_generater import Generator

device = torch.device("cuda")

"""
Infer programs on cad dataset
"""
def infer_programs(imitate_net, path):

    config = read_config.Config("config_rl.yml")
    max_len = 13
    beam_width = 10

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

    generator = Generator()

    train_gen = generator.train_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    Rs = 0
    CDs = 0
    Target_images = []
    pred_images = np.zeros((config.train_size, 64, 64))
    for batch_idx in range(config.train_size // config.batch_size):
        with torch.no_grad():
            print(f"Inferring cad batch: {batch_idx}")
            data_ = next(train_gen)
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

            # select best expression by chamfer distance
            best_labels = np.zeros((config.batch_size, max_len))
            for r in range(config.batch_size):
                idx = np.argmin(beam_CD[r * beam_width:(r + 1) * beam_width])
                best_labels[r] = beam_labels[r][idx]
            pred_labels[batch_idx*config.batch_size:batch_idx*config.batch_size + config.batch_size] = best_labels

            CD = np.zeros((config.batch_size, 1))
            for r in range(config.batch_size):
                CD[r, 0] = min(beam_CD[r * beam_width:(r + 1) * beam_width])
                pred_images[batch_idx*config.batch_size+r] = predicted_images[r*beam_width + np.argmin(beam_CD[r*beam_width:(r+1)*beam_width])]

            CDs += np.mean(CD)

    print(f"TRAIN CD: {CDs / (config.train_size // config.batch_size)}")

    torch.save(pred_labels, path + "labels.pt")
    torch.save(pred_images, path + "images.pt")

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

    print(f"TEST CD: {CDs / (config.test_size // config.batch_size)}")
