from __future__ import print_function
import argparse
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
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer

VOCAB_SIZE = 400

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.embed = nn.Linear(VOCAB_SIZE, 100)
        self.encode_gru = nn.GRU(100, 256)
        self.fc11 = nn.Linear(256, 20)
        self.fc12 = nn.Linear(256, 20)
        self.decode_gru = nn.GRU(20, 100)
        self.unembed = nn.Linear(100, VOCAB_SIZE)

    def encode(self, x):
        embeddings = self.embed(x)
        _, h = self.encode_gru(embeddings)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, timesteps):
        # repeat latent code to create sequence to feed to decoder
        output, h = self.decode_gru(z.repeat(timesteps, 1, 1))
        return self.unembed(output)

    def forward(self, x):
        # shape of x should be (timesteps, batch_size, features)
        x = x.permute(1, 0, 2)
        timesteps = x.shape[0]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, timesteps), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        # target needs to be (batch_size, timesteps)
        target = x.argmax(dim=2)
        # output needs to be batch_size, features, timesteps)
        recon_x = recon_x.permute(1, 2, 0)
        BCE = F.cross_entropy(recon_x, target, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
