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

max_len = 13
beam_width = 10

"""
Infer programs on cad dataset
"""
def infer_programs(imitate_net, path, num_train, num_test, BATCH_SIZE, self_training):    
    config = read_config.Config("config_cad.yml")

    # Load the terminals symbols of the grammar
    with open("terminals.txt", "r") as file:
        unique_draw = file.readlines()
    for index, e in enumerate(unique_draw):
        unique_draw[index] = e[0:-1]

    config.train_size = num_train
    config.test_size = num_test
    config.batch_size = BATCH_SIZE
    
    imitate_net.eval()
    imitate_net.epsilon = 0
    
    parser = ParseModelOutput(unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
        
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

    test_gen =  generator.test_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)
                                   
    start = time.time()

    pred_labels = np.zeros((config.train_size, max_len))
    Target_images = []

    results = {}
    
    for _gen, name, do_write, iters in [
            (train_gen, 'train', True,  config.train_size // config.batch_size),
            (val_gen, 'val', False, config.test_size // config.batch_size),
            (test_gen, 'test', False, config.test_size // config.batch_size),
    ]:
        
        print(f"Inferring cad for {name}")
        CDs = 0.
        count = 0.
        save_viz = True
        
        for batch_idx in range(iters):
            with torch.no_grad():                
                data_ = next(_gen)
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
                if do_write:
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

                predicted_images = image_from_expressions(parser, expressions)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
                target_images_new = np.repeat(
                    target_images, axis=0, repeats=beam_width)

                beam_CD = chamfer(target_images_new, predicted_images)

                if do_write:
                    best_labels = np.zeros((config.batch_size, max_len))
                    for r in range(config.batch_size):
                        idx = np.argmin(beam_CD[r * beam_width:(r + 1) * beam_width])
                        best_labels[r] = beam_labels[r][idx]
                    pred_labels[batch_idx*config.batch_size:batch_idx*config.batch_size + config.batch_size] = best_labels

            CD = np.zeros((config.batch_size, 1))
            for r in range(config.batch_size):
                CD[r, 0] = min(beam_CD[r * beam_width:(r + 1) * beam_width])

            CDs += np.mean(CD)
            count += 1
                        
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
                        f"{name}_{batch_idx * config.batch_size + j}.png",
                        transparent=0)
                    plt.close("all")
                    save_viz = False

        avg_cd = CDs / count
        print(f"AVG CD for {name}: {avg_cd}")
        results[name] = avg_cd
        
    with open(results_path + "results_beam_width_{}.org".format(beam_width),
              'w') as outfile:
        json.dump(results, outfile)

    torch.save(pred_labels, labels_path + "labels.pt")
    if self_training:
        torch.save(np.concatenate(Target_images, axis=0), labels_path + "images.pt")

    end = time.time()
    print(f"Inference time: {end-start}")

    return results
