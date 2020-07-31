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
from globals import device

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, vocab_size):
        super(VAE, self).__init__()

        self.relu = nn.ReLU()
        self.encode_embed = nn.Embedding(vocab_size, hidden_dim)
        self.decode_embed = nn.Embedding(vocab_size, hidden_dim)
        self.encode_gru = nn.GRU(hidden_dim, hidden_dim)
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decode_gru = nn.GRU(latent_dim+hidden_dim, hidden_dim)
        #self.decode_gru = nn.GRU(latent_dim, hidden_dim)
        self.dense_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, vocab_size)
        self.initial_encoder_state = nn.Parameter(torch.randn((1, 1, hidden_dim)))
        self.initial_decoder_state = nn.Parameter(torch.randn((1, 1, hidden_dim)))

    def encode(self, encoder_input):
        init_state = self.initial_encoder_state.repeat(1, encoder_input.shape[1], 1)
        _, h = self.encode_gru(self.relu(self.encode_embed(encoder_input)), init_state)
        return self.encode_mu(h), self.encode_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, decoder_input=None, timesteps=None):
        init_state = self.initial_decoder_state.repeat(1, z.shape[1], 1)
        output, h = self.decode_gru(z.repeat(timesteps, 1, 1), init_state)
        return self.dense_2(self.relu(self.dense_1(output)))
        # training
        # batch_size = z.shape[1]
        # if decoder_input is not None:
        #     # concatentate latent code with input sequence
        #     comb_input = torch.cat([z.repeat(timesteps, 1, 1), self.relu(self.decode_embed(decoder_input))], 2)
        #     init_state = self.initial_decoder_state.repeat(1, batch_size, 1)
        #     output, h = self.decode_gru(comb_input, init_state)
        #     return self.dense_2(self.relu(self.dense_1(output)))
        # # sampling
        # else:
        #     z_seq = torch.reshape(z, (1, batch_size, z.shape[2]))
        #     # initial token is the start/stop token (399)
        #     output_token = torch.full((1, batch_size), 399, dtype=torch.long).to(device).long()
        #     h = self.initial_decoder_state.repeat(1, batch_size, 1)
        #
        #     output_list = []
        #     # loop through sequence using decoded output (plus latent code) and hidden state as next input
        #     for _ in range(timesteps-1):
        #         in_seq = self.relu(self.decode_embed(output_token)).view(1, batch_size, -1)
        #         output, h = self.decode_gru(torch.cat([z_seq, in_seq], 2), h)
        #         output = self.dense_2(self.relu(self.dense_1(output[0])))
        #         output_list.append(output)
        #         output_token = torch.argmax(output, dim=1)
        #     return torch.stack(output_list)

    def forward(self, x):
        # shape of x should be (timesteps, batch_size)
        x = x.permute(1, 0)
        # remove stop token
        decoder_input = x[:-1, :]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, decoder_input, decoder_input.shape[0]), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        # remove start token from labels
        x = x[:, 1:]
        # for cross entropy output needs to be batch_size, features, timesteps)
        recon_x = recon_x.permute(1, 2, 0)
        BCE = F.cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    # def clean(self, prev_output, next_output, timesteps):
    #     """
    #     Zeros logits in next output that would create an invalid program
    #     reruns gru
    #     """
    #     prev_seq = torch.argmax(self.dense(prev_output), dim=2)
    #     cleaned = np.zeros(next_output.shape)
    #
    #     # maybe a way to batch this but unsure at the moment
    #     for j in range(prev_seq.shape[1])
    #         seq = prev_seq[:, j]
    #
    #         num_draws = 0
    #         num_ops = 0
    #         for i, t in enumerate(seq):
    #             if t < 396:
    #                 # draw a shape on canvas kind of operation
    #                 num_draws += 1
    #             elif t >= 396 and t < 399:
    #                 # +, *, - kind of operation
    #                 num_ops += 1
    #             elif t == 399:
    #                 # Stop symbol, no need to process further
    #                 if num_draws > ((len(seq) - 1) // 2 + 1):
    #                     return False
    #                 if not (num_draws > num_ops):
    #                     return False
    #                 return (num_draws - 1) == num_ops
    #
    #             if num_draws <= num_ops:
    #                 # condition where number of operands are lesser than 2
    #                 return False
    #             if num_draws > (timesteps // 2 + 1):
    #                 # condition for stack over flow
    #                 return False
    #
    #         return (num_draws - 1) == num_ops
