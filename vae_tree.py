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
    def __init__(self, hidden_dim, latent_dim, vocab_size, max_len):
        super(VAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.parent = nn.Linear(3*hidden_dim, hidden_dim)
        self.child1 = nn.Linear(latent_dim, hidden_dim)
        self.child2 = nn.Linear(hidden_dim, hidden_dim+1)
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.decode_leaf = nn.Linear(hidden_dim, vocab_size)
        self.decode_parent = nn.Linear(hidden_dim, 2*latent_dim+vocab_size)


    def encode(self, tree):
        def traverse(node):
            if node["right"] is None and node["left"] is None:
                return self.embed(node["value"])
            else:
                lchild = traverse(node["left"])
                rchild = traverse(node["right"])
                par = self.embed(node["value"])
                input = torch.cat([par, lchild, rchild], 0)
                return self.parent(input)

        h = traverse(tree)
        return self.encode_mu(h), self.encode_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        def traverse(inp):
            final = []
            stack = []
            stack.append(inp)
            while not (stack == []):
                node_latent = stack.pop()
                child_out = self.child2(self.relu(self.child1(node_latent)))
                # not a leaf
                if child_out[0] > 0 and len(stack) <= self.max_len-2:
                    decoded = self.decode_parent(child_out[1:])
                    value = decoded[:self.vocab_size]
                    llatent = decoded[self.vocab_size:self.vocab_size+self.latent_dim]
                    rlatent = decoded[self.vocab_size+self.latent_dim:]
                    stack.append(llatent)
                    stack.append(rlatent)
                    final.append(value)
                # a leaf
                else:
                    value = self.decode_leaf(child_out[1:])
                    final.append(value)
            return final

        # UNFINISHED DOES NOT WORK

        return traverse(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        def flatten(node):
            if node["right"] is None and node["left"] is None:
                return [node["value"]]
            else:
                lchild = flatten(node["left"])
                rchild = flatten(node["right"])
                return [node["value"]] + lchild + rchild

        flat_x = torch.stack(flatten(x))
        flat_recon_x = torch.stack(flatten(recon_x))
        print(flat_recon_x.shape)
        flat_x = F.pad(flat_x, (0, self.max_len-len(flat_x)), 'constant', 399)
        flat_recon_x = F.pad(flat_recon_x, (0, 0, 0, self.max_len-len(flat_recon_x)), 'constant', 399)

        CE = F.cross_entropy(flat_recon_x, flat_x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return CE + KLD
