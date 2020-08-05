import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from globals import device

class MLP(nn.Module):
    def __init__(self, ind, hdim, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, odim)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

# VAE for trees represented as dicts with "value", "left", and "right" keys
class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, vocab_size, max_len):
        super(VAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0)[0], requires_grad=True)

        # encoding
        self.embed = nn.Embedding(self.vocab_size, hidden_dim)
        # self.encode_union = MLP(3*hidden_dim, hidden_dim, hidden_dim)
        # self.encode_intersect = MLP(3*hidden_dim, hidden_dim, hidden_dim)
        # self.encode_subtract = MLP(3*hidden_dim, hidden_dim, hidden_dim)
        # self.encode_mu = MLP(hidden_dim, hidden_dim, latent_dim)
        # self.encode_logvar = MLP(hidden_dim, hidden_dim, latent_dim)
        #
        # # decoding
        # self.node_type = MLP(latent_dim, hidden_dim, self.vocab_size)
        # self.decode_union = MLP(latent_dim, hidden_dim, 2*latent_dim)
        # self.decode_intersect = MLP(latent_dim, hidden_dim, 2*latent_dim)
        # self.decode_subtract = MLP(latent_dim, hidden_dim, 2*latent_dim)
        self.encode_union = nn.Linear(2*hidden_dim, hidden_dim)
        self.encode_intersect = nn.Linear(2*hidden_dim, hidden_dim)
        self.encode_subtract = nn.Linear(2*hidden_dim, hidden_dim)
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoding
        self.node_type = nn.Linear(latent_dim, self.vocab_size)
        self.decode_union = nn.Linear(latent_dim, 2*latent_dim)
        self.decode_intersect = nn.Linear(latent_dim, 2*latent_dim)
        self.decode_subtract = nn.Linear(latent_dim, 2*latent_dim)

    def encode(self, tree):
        def traverse(node):
            # leaf
            if node["right"] is None and node["left"] is None:
                return torch.tanh(self.embed(node["value"]))
            # internal
            else:
                lchild = traverse(node["left"])
                rchild = traverse(node["right"])
                input = torch.cat([lchild, rchild], 0)
                if node["value"] == 396:
                    return torch.tanh(self.encode_union(input))
                elif node["value"] == 397:
                    return torch.tanh(self.encode_intersect(input))
                elif node["value"] == 398:
                    return torch.tanh(self.encode_subtract(input))
                else:
                    assert(False)

        h = traverse(tree)
        return self.encode_mu(h), self.encode_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, tree=None):
        # returns decoded tree and a list of predicted node types
        # (for training node_type classifier), teacher forces tree structure
        def traverse_train(node, code):
            # leaf
            if node["right"] is None and node["left"] is None:
                return {"value": self.node_type(code), "left": None, "right": None}
            # internal
            else:
                if node["value"] == 396:
                    par_out = torch.tanh(self.decode_union(code))
                elif node["value"] == 397:
                    par_out = torch.tanh(self.decode_intersect(code))
                elif node["value"] == 398:
                    par_out = torch.tanh(self.decode_subtract(code))
                else:
                    assert(False)
                lchild = traverse_train(node["left"], par_out[:self.latent_dim])
                rchild = traverse_train(node["right"], par_out[self.latent_dim:])
                return {"value": self.node_type(code), "left": lchild, "right": rchild}

        # returns decoded tree given just a latent code at test time
        def traverse_test(code, max_depth):
            type = self.node_type(code)
            token = torch.argmax(type)
            # leaf
            if token < 396 or max_depth == 1:
                zeroed_type = type
                zeroed_type[396:] = 0
                return {"value": zeroed_type, "left": None, "right": None}
            # internal
            else:
                if token == 396:
                    par_out = torch.tanh(self.decode_union(code))
                elif token == 397:
                    par_out = torch.tanh(self.decode_intersect(code))
                elif token == 398:
                    par_out = torch.tanh(self.decode_subtract(code))
                else:
                    assert(False)
                lchild = traverse_test(par_out[:self.latent_dim], max_depth - 1)
                rchild = traverse_test(par_out[self.latent_dim:], max_depth - 1)
                return {"value": type, "left": lchild, "right": rchild}

        if tree is not None:
            return traverse_train(tree, z)
        else:
            # depth of 7 is enough to generate max_len=13 programs
            # todo: don't hardcode this
            return traverse_test(z, 7)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x), mu, logvar

    # Reconstruction + KL divergence losses
    def loss_function(self, recon_x, x, mu, logvar, beta):
        # returns flattened tree and target node types
        def flatten(node):
            if node["right"] is None and node["left"] is None:
                return [node["value"]]
            else:
                lchild = flatten(node["left"])
                rchild = flatten(node["right"])
                return [node["value"]] + lchild + rchild

        flat_x = flatten(x)
        flat_recon_x = flatten(recon_x)
        # print(torch.stack(flat_x))
        # print(torch.argmax(torch.stack(flat_recon_x), dim=1))

        CE = F.cross_entropy(torch.stack(flat_recon_x), torch.stack(flat_x), reduction='sum')
        #CE = gaussian_nll(torch.stack(flat_recon_x), softclip(self.log_sigma, -6), F.one_hot(torch.stack(flat_x), 399)).sum()

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # KLD *= .01
        # print(CE/KLD)
        return CE + beta * KLD, CE, KLD

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def softclip(tensor, min):
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor
