import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from globals import device

class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)

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

        # encoding
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.parent = MLP(3*hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.encode_mu = nn.Linear(hidden_dim, latent_dim)
        self.encode_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoding
        self.nodetype = MLP(latent_dim, hidden_dim, hidden_dim, 1) # classifies nodes as leaf or internal
        self.decode_leaf = MLP(latent_dim, hidden_dim, hidden_dim, vocab_size)
        self.decode_parent = MLP(latent_dim, hidden_dim, hidden_dim, 2*latent_dim+vocab_size)

    def encode(self, tree):
        def traverse(node):
            # leaf
            if node["right"] is None and node["left"] is None:
                return self.embed(node["value"])
            # internal
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

    def decode(self, z, tree=None):
        # returns decoded tree and a list of predicted node types
        # (for training nodetype classifier), teacher forces tree structure
        def traverse_train(node, code):
            # leaf
            if node["right"] is None and node["left"] is None:
                return {"value": self.decode_leaf(code), "left": None, "right": None}, [self.nodetype(code)]
            # internal
            else:
                par_out = self.decode_parent(code)
                lchild, ltype = traverse_train(node["left"], par_out[:self.latent_dim])
                rchild, rtype = traverse_train(node["right"], par_out[self.latent_dim:2*self.latent_dim])
                return {"value": par_out[2*self.latent_dim:], "left": lchild, "right": rchild}, [self.nodetype(code)] + ltype + rtype

        # returns decoded tree given just a latent code at test time
        def traverse_test(code, max_depth):
            # leaf
            if self.nodetype(code) < 0 or max_depth == 1:
                return {"value": self.decode_leaf(code), "left": None, "right": None}
            # internal
            else:
                par_out = self.decode_parent(code)
                lchild = traverse_test(par_out[:self.latent_dim], max_depth - 1)
                rchild = traverse_test(par_out[self.latent_dim:2*self.latent_dim], max_depth - 1)
                return {"value": par_out[2*self.latent_dim:], "left": lchild, "right": rchild}

        if tree is not None:
            return traverse_train(tree, z)
        else:
            # depth of 4 is enough to generate max_len=13 programs
            # todo: don't hardcode this
            return traverse_test(z, 4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x), mu, logvar

    # Reconstruction + KL divergence losses
    def loss_function(self, decoder_out, x, mu, logvar):
        # returns flattened tree and target node types
        def flatten(node):
            if node["right"] is None and node["left"] is None:
                return [node["value"]], [torch.tensor([0])]
            else:
                lchild, ltype = flatten(node["left"])
                rchild, rtype = flatten(node["right"])
                return [node["value"]] + lchild + rchild, [torch.tensor([1])] + ltype + rtype

        recon_x, type_list = decoder_out
        flat_x, target_type_list = flatten(x)
        flat_recon_x, _ = flatten(recon_x)

        CE = F.cross_entropy(torch.stack(flat_recon_x), torch.stack(flat_x), reduction='sum')
        # loss from predicting node type
        type_loss = F.binary_cross_entropy_with_logits(torch.stack(type_list), torch.stack(target_type_list).float().to(device), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # probably should rescale these componenents
        return CE + type_loss + KLD
