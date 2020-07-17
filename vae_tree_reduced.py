import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from globals import device

# VAE for trees represented as dicts with "value", "left", and "right" keys
class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, vocab_size, max_len):
        super(VAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.relu = torch.nn.LeakyReLU()

        # encoding
        self.embed = nn.Embedding(self.vocab_size, hidden_dim)
        self.parent = nn.Linear(3*hidden_dim, hidden_dim) # maybe make this different depending on the operator or a mlp
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoding
        self.nodetype = nn.Linear(latent_dim, self.vocab_size)
        self.decode_union = nn.Linear(latent_dim, 2*latent_dim)
        self.decode_intersect = nn.Linear(latent_dim, 2*latent_dim)
        self.decode_subtract = nn.Linear(latent_dim, 2*latent_dim)

    def encode(self, tree):
        def traverse(node):
            # leaf
            if node["right"] is None and node["left"] is None:
                return self.relu(self.embed(node["value"]))
            # internal
            else:
                lchild = traverse(node["left"])
                rchild = traverse(node["right"])
                par = self.relu(self.embed(node["value"]))
                input = torch.cat([par, lchild, rchild], 0)
                return self.relu(self.parent(input))

        h = traverse(tree)
        return self.encode_mu(h), self.encode_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, tree=None):
        # returns decoded tree and a list of predicted node types
        # (for training nodetype classifier), teacher forces tree structure
        def traverse_train(node, code):
            # leaf
            if node["right"] is None and node["left"] is None:
                return {"value": self.nodetype(code), "left": None, "right": None}
            # internal
            else:
                if node["value"] == 396:
                    par_out = self.relu(self.decode_union(code))
                elif node["value"] == 397:
                    par_out = self.relu(self.decode_intersect(code))
                elif node["value"] == 398:
                    par_out = self.relu(self.decode_subtract(code))
                else:
                    assert(False)
                lchild = traverse_train(node["left"], par_out[:self.latent_dim])
                rchild = traverse_train(node["right"], par_out[self.latent_dim:])
                return {"value": self.nodetype(code), "left": lchild, "right": rchild}

        # returns decoded tree given just a latent code at test time
        def traverse_test(code, max_depth):
            type = self.nodetype(code)
            token = torch.argmax(type)
            # leaf
            if token < 396 or max_depth == 1:
                return {"value": type, "left": None, "right": None}
            # internal
            else:
                if token == 396:
                    par_out = self.relu(self.decode_union(code))
                elif token == 397:
                    par_out = self.relu(self.decode_intersect(code))
                elif token == 398:
                    par_out = self.relu(self.decode_subtract(code))
                else:
                    assert(False)
                lchild = traverse_test(par_out[:self.latent_dim], max_depth - 1)
                rchild = traverse_test(par_out[self.latent_dim:], max_depth - 1)
                return {"value": type, "left": lchild, "right": rchild}

        if tree is not None:
            return traverse_train(tree, z)
        else:
            # depth of 4 is enough to generate max_len=13 programs
            # todo: don't hardcode this
            return traverse_test(z, 5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x), mu, logvar

    # Reconstruction + KL divergence losses
    def loss_function(self, recon_x, x, mu, logvar):
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

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # KLD *= .01
        # print(CE/KLD)
        return CE + KLD
