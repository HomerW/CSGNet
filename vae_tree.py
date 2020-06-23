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

device = torch.device("cpu")

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, vocab_size):
        super(VAE, self).__init__()

        self.encode_embed = nn.Embedding(vocab_size, hidden_dim)
        self.decode_embed = nn.Embedding(vocab_size, hidden_dim)
        self.leaf = nn.Linear(hidden_dim, hidden_dim)
        self.parent = nn.Linear(3*hidden_dim, hidden_dim)
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(self, encoder_input):


        return self.encode_mu(h), self.encode_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        pass

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        # for cross entropy output needs to be batch_size, features, timesteps)
        recon_x = recon_x.permute(1, 2, 0)
        BCE = F.cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
