import numpy as np
import torch
from src.utils.train_utils import chamfer
from src.utils.generators.shapenet_generater import Generator

def get_nn(images1, images2):
    min_cds = []
    for i in range(len(images1)):
        repeated_i = np.repeat(images1[i:i+1], axis=0, repeats=len(images2))
        cd = chamfer(repeated_i, images2)
        min_cds.append(np.amin(cd))
    return sum(min_cds)/len(min_cds)

def get_bidir_nn(images1, images2):
    return get_nn(images1, images2) + get_nn(images2, images1)
