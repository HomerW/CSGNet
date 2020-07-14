import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from src.utils.generators.shapenet_generater import Generator
from vae import VAE
from ws_infer import infer_programs
from vae_tree_reduced import VAE
from tree_conversion import label_to_tree, tree_to_label
from ws_infer import infer_programs
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.generators.wake_sleep_gen import WakeSleepGen

class FIDModel(nn.Module):
    def __init__(self):
        super(FIDModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.dense = nn.Linear(2048, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.dense(x.view((batch_size, -1)))
        return x

    def encode(self, x):
        batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def loss_function(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

if __name__ == '__main__':
    device = torch.device("cuda")
    inference_train_size = 10000
    inference_test_size = 3000
    vocab_size = 400
    generator_hidden_dim = 256
    generator_latent_dim = 20
    max_len = 13
    batch_size = 100
    real_batch_size = batch_size // 2
    epochs = 50

    data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                         5: "data/synthetic/two_ops/expressions.txt",
                         7: "data/synthetic/three_ops/expressions.txt",
                         9: "data/synthetic/four_ops/expressions.txt",
                         11: "data/synthetic/five_ops/expressions.txt",
                         13: "data/synthetic/six_ops/expressions.txt"}
    dataset_sizes = {
        3: [30000, 5000],
        5: [110000, 50000],
        7: [170000, 50000],
        9: [270000, 50000],
        11: [370000, 100000],
        13: [370000, 100000]
    }
    dataset_sizes = {k: [x // 100 for x in v] for k, v in dataset_sizes.items()}
    syn_batch_size = (batch_size // 2) // len(dataset_sizes)
    syn_gen = MixedGenerateData(data_labels_paths=data_labels_paths,
                                batch_size=syn_batch_size)
    syn_gen_train = {}
    syn_gen_test = {}
    for k in data_labels_paths.keys():
        syn_gen_train[k] = syn_gen.get_train_data(
            syn_batch_size,
            k,
            num_train_images=dataset_sizes[k][0],
            jitter_program=True)
        syn_gen_test[k] = syn_gen.get_test_data(
            syn_batch_size,
            k,
            num_train_images=dataset_sizes[k][0],
            num_test_images=dataset_sizes[k][1],
            jitter_program=True)

    def get_syn_batch(gen):
        sub_batches = []
        for k in dataset_sizes.keys():
            sub_batches.append(torch.from_numpy(next(gen[k])[0][-1, :, 0:1, :, :]).to(device))
        return torch.cat(sub_batches)

    real_gen = WakeSleepGen(f"wake_sleep_data/best_labels_full/labels.pt",
                            f"wake_sleep_data/best_labels_full/val/labels.pt",
                            batch_size=real_batch_size,
                            train_size=inference_train_size,
                            test_size=inference_test_size)
    real_gen_train = real_gen.get_train_data()
    real_gen_test = real_gen.get_test_data()

    model = FIDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    actual_batch_size = (real_batch_size + syn_batch_size * 6)
    labels = torch.cat([torch.ones((actual_batch_size  // 2)), torch.zeros((actual_batch_size // 2))])
    labels = labels.to(device).reshape((labels.shape[0], 1))

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx in range(inference_train_size // batch_size):
            optimizer.zero_grad()
            real_batch = next(real_gen_train)[0][-1, :, 0:1, :, :].to(device)
            syn_batch = get_syn_batch(syn_gen_train)
            # each batch is 1/2 synthetic, 1/2 real inferred programs
            batch = torch.cat([real_batch, syn_batch])
            logits = model(batch)
            loss = model.loss_function(logits, labels)
            train_loss += float(loss)
            #print(f"epoch {epoch}, batch {batch_idx}, train loss {loss.data}")
            loss.backward()
            optimizer.step()
        print(f"average train loss {epoch}: {train_loss / (inference_train_size // batch_size)}")
        test_loss = 0
        for batch_idx in range(inference_test_size // batch_size):
            with torch.no_grad():
                real_batch = next(real_gen_test)[0][-1, :, 0:1, :, :].to(device)
                syn_batch = get_syn_batch(syn_gen_test)
                # each batch is 1/2 synthetic, 1/2 real inferred programs
                batch = torch.cat([real_batch, syn_batch])
                logits = model(batch)
                loss = model.loss_function(logits, labels)
                test_loss += float(loss)
        print(f"average test loss {epoch}: {test_loss / (inference_test_size // batch_size)}")

    torch.save(model.state_dict(), f"trained_models/fid-model.pth")
