#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from src.utils import read_config
from src.Models.models import ImitateJoint
import matplotlib.pyplot as plt
from torch.nn import functional as F
from src.Models.models import ImitateJoint, ParseModelOutput
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer, beams_parser, validity, image_from_expressions, stack_from_expressions
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.generators.shapenet_generater import Generator

class FidGen:
    def __init__(self, images_path):
        self.images = torch.load(images_path)
    def get_test_data(self):
        while True:
            for i in range(0, 3000, 100):
                batch_images = self.images[i:i+100]
                yield batch_images

def get_nn(images1, images2):
    min_cds = []
    for i in range(len(images1)):
        repeated_i = np.repeat(images1[i:i+1], axis=0, repeats=len(images2))
        cd = chamfer(repeated_i, images2)
        min_cds.append(np.amin(cd))
    return sum(min_cds)/len(min_cds)

def get_bidir_nn(images1, images2):
    return get_nn(images1, images2) + get_nn(images2, images1)


def get_all_cad():
    cad_generator = Generator().test_gen(batch_size=100,
                                        path="data/cad/cad.h5",
                                        if_augment=False)
    images = np.zeros((3000, 64, 64))
    for i in range(3000 // 100):
        images[i*100:i*100+100] = next(cad_generator)[-1, :, 0, :, :]
    return images[:500]

def get_all_val():
    cad_generator = Generator().val_gen(batch_size=100,
                                        path="data/cad/cad.h5",
                                        if_augment=False)
    images = np.zeros((3000, 64, 64))
    for i in range(3000 // 100):
        images[i*100:i*100+100] = next(cad_generator)[-1, :, 0, :, :]
    return images[:500]

if __name__ == '__main__':
    cad_images = get_all_cad()
    val_images = get_all_val()
    # random_images = torch.load("random_images.pt")[:500]
    distance = get_bidir_nn(cad_images, val_images)
    print(distance)
    # distances = []
    # for i in range(39):
    #     lest_images = torch.load(f"fid_images2/{i}.pt")[:500]
    #     cad_images = get_all_cad()
    #     distance = get_bidir_nn(random_images, cad_images)
    #     distances.append(distance)
    #     print(distance)
    # with open("distances.txt", "w") as file:
    #     for d in distances:
    #         file.write(f"{d}\n")
