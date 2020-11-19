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
from globals import device
from autoencoder import Autoencoder
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from src.Models.models import Encoder
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.generators.shapenet_generater import Generator
from globals import device

def get_activations(generator, model, batch_size=50, dims=32, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    test_size = 3000

    pred_arr = np.empty((test_size, dims))

    n_batches = test_size // batch_size

    for i in tqdm(range(0, test_size, batch_size)):
        if verbose:
            print('\rPropagating batch %d/%d' % ((i + 1) // batch_size, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        images = next(generator)
        images = torch.from_numpy(images).to(device)
        if len(images.shape) == 3: # generated samples
            # images = torch.from_numpy(np.random.randint(0, 2, (100, 64, 64))).to(device).float()
            pred = model.encode(images.unsqueeze(1).float())
        else: # cad data
            pred = model.encode(images[-1, :, 0:1, :, :])

        pred = pred.reshape((batch_size,-1, 1, 1))
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=32, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        m, s = calculate_activation_statistics(path, model, batch_size,
                                               dims)

    return m, s

def calculate_fid_given_paths(model, images_path, model_path, batch_size, dims=32):
    """Calculates the FID of two paths"""
    if not os.path.exists(images_path):
        raise RuntimeError('Invalid path: %s' % images_path)
    if not os.path.exists(model_path):
        raise RuntimeError('Invalid path: %s' % model_path)

    generator = FidGen(images_path).get_test_data()
    # generator2 = FidGen(model_path).get_test_data()
    cad_generator = Generator().test_gen(batch_size=batch_size,
                                        path="data/cad/cad.h5",
                                        if_augment=False)
    # cad_generator2 = Generator().val_gen(batch_size=batch_size,
    #                                     path="data/cad/cad.h5",
    #                                     if_augment=False)

    m1, s1 = calculate_activation_statistics(cad_generator, model, batch_size,
                                         dims)
    m2, s2 = calculate_activation_statistics(generator, model, batch_size,
                                         dims)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def get_csgnet():
    config = read_config.Config("config_synthetic.yml")

    # Encoder
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

    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath, map_location=device)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    return imitate_net.encoder

class FidGen:
    def __init__(self, images_path):
        self.images = torch.load(images_path)
    def get_test_data(self):
        while True:
            for i in range(0, 3000, 100):
                batch_images = self.images[i:i+100]
                yield batch_images

if __name__ == '__main__':
    # model = Autoencoder().to(device)
    # model.load_state_dict(torch.load("trained_models/fid-model2.pth"))
    model = get_csgnet()
    fids = []
    fid_value = calculate_fid_given_paths(model,
                                          f"fid_images2/base.pt",
                                          "trained_models/mix_len_cr_percent_equal_batch_3_13_prop_100_hdsz_2048_batch_2000_optim_adam_lr_0.001_wd_0.0_enocoderdrop_0.0_drop_0.2_step_mix_mode_12.pth",
                                          # "random_images.pt",
                                          100,
                                          2048)
    print(fid_value)
    # for i in range(17):
    #     fid_value = calculate_fid_given_paths(model,
    #                                           f"fid_images2/{i}.pt",
    #                                           # "trained_models/mix_len_cr_percent_equal_batch_3_13_prop_100_hdsz_2048_batch_2000_optim_adam_lr_0.001_wd_0.0_enocoderdrop_0.0_drop_0.2_step_mix_mode_12.pth",
    #                                           "random_images.pt",
    #                                           100,
    #                                           2048)
    #
    #     print('FID: ', fid_value)
    #     fids.append(fid_value)

    # fig, ax = plt.subplots()
    # ax.plot(fids)
    # plt.savefig("fids.png")
    # with open("fids_random.txt", "w") as file:
    #     for f in fids:
