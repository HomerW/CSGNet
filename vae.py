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

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# trains for 1 epoch
def train(input):
    model.train()
    train_loss = 0
    np.random.shuffle(input)
    for i in range(0, len(input), config.batch_size):
        batch = torch.stack(input[i:i+config.batch_size], dim=0)
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = model.loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"loss: {train_loss / (len(input) // config.batch_size)}")

config = read_config.Config("config_synthetic.yml")

data_labels_paths = {
    3: "data/synthetic/one_op/expressions.txt",
    # 5: "data/synthetic/two_ops/expressions.txt",
    # 7: "data/synthetic/three_ops/expressions.txt"
}

# proportion is in percentage. vary from [1, 100].
proportion = config.proportion
dataset_sizes = {
    3: [proportion * 250, proportion * 50],
    # 5: [proportion * 1000, proportion * 100],
    # 7: [proportion * 1500, proportion * 200]
}

generator = MixedGenerateData(
    data_labels_paths=data_labels_paths,
    batch_size=config.batch_size,
    canvas_shape=config.canvas_shape)

max_len = max(data_labels_paths.keys())

types_prog = len(dataset_sizes)
train_gen_objs = {}
test_gen_objs = {}
config.train_size = sum(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = sum(dataset_sizes[k][1] for k in dataset_sizes.keys())
total_importance = sum(k for k in dataset_sizes.keys())
for k in data_labels_paths.keys():
    test_batch_size = int(config.batch_size * dataset_sizes[k][1] / \
                          config.test_size)
    # Acts as a curriculum learning
    train_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(
        train_batch_size,
        k,
        num_train_images=dataset_sizes[k][0],
        jitter_program=True)
    test_gen_objs[k] = generator.get_test_data(
        test_batch_size,
        k,
        num_train_images=dataset_sizes[k][0],
        num_test_images=dataset_sizes[k][1],
        jitter_program=True)

data = []
for batch_idx in range(config.test_size // (config.batch_size)):
    _, labels = next(train_gen_objs[3])
    one_hot_labels = prepare_input_op(labels,
                                      len(generator.unique_draw))
    one_hot_labels = Variable(
        torch.from_numpy(one_hot_labels))
    labels = Variable(torch.from_numpy(labels))
    data += (one_hot_labels)

for _ in range(100):
    train(data)
