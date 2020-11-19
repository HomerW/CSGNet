import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from src.utils.generators.shapenet_generater import Generator
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.generators.shapenet_generater import Generator
from globals import device

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))

        self.convt1 = nn.ConvTranspose2d(32, 16, 3, padding=(1, 1))
        self.convt2 = nn.ConvTranspose2d(16, 8, 3, padding=(1, 1))
        self.convt3 = nn.ConvTranspose2d(8, 1, 3, padding=(1, 1))
        self.dense = nn.Linear(2048, 1)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def decode(self, x):
        x = F.interpolate(F.relu(self.convt1(x)), scale_factor=(2, 2))
        x = F.interpolate(F.relu(self.convt2(x)), scale_factor=(2, 2))
        x = F.interpolate(torch.sigmoid(self.convt3(x)), scale_factor=(2, 2))
        return x

    def loss_function(self, logits, labels):
        return F.binary_cross_entropy(logits, labels)

if __name__ == '__main__':
    vocab_size = 400
    batch_size = 300
    epochs = 1000

    data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                         5: "data/synthetic/two_ops/expressions.txt",
                         7: "data/synthetic/three_ops/expressions.txt",
                         9: "data/synthetic/four_ops/expressions.txt",
                         11: "data/synthetic/five_ops/expressions.txt",
                         13: "data/synthetic/six_ops/expressions.txt"}
    dataset_sizes = {
        3: [1000, 5000],
        5: [1000, 50000],
        7: [1500, 50000],
        9: [2500, 50000],
        11: [3500, 100000],
        13: [3500, 100000]
    }
    inference_train_size = sum([v[0] for k, v in dataset_sizes.items()])
    syn_gen = MixedGenerateData(data_labels_paths=data_labels_paths,
                                batch_size=batch_size)
    syn_gen_train = {}
    syn_gen_test = {}
    for k in data_labels_paths.keys():
        syn_gen_train[k] = syn_gen.get_train_data(
            batch_size // 6,
            k,
            num_train_images=dataset_sizes[k][0],
            jitter_program=True)
        syn_gen_test[k] = syn_gen.get_test_data(
            batch_size // 6,
            k,
            num_train_images=dataset_sizes[k][0],
            num_test_images=dataset_sizes[k][1],
            jitter_program=True)

    def get_syn_batch(gen):
        sub_batches = []
        for k in dataset_sizes.keys():
            sub_batches.append(torch.from_numpy(next(gen[k])[0][-1, :, 0:1, :, :]).to(device))
        return torch.cat(sub_batches)

    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    random_images = np.zeros((3000, 64, 64))
    for epoch in range(10):
        random_images[epoch*batch_size:epoch*batch_size + batch_size] = get_syn_batch(syn_gen_train).squeeze().cpu().numpy()
        # model.train()
        # train_loss = 0
        # acc = 0
        # for batch_idx in range(inference_train_size // batch_size):
        #     optimizer.zero_grad()
        #     batch = get_syn_batch(syn_gen_train)
        #     logits = model(batch)
        #     loss = model.loss_function(logits, batch)
        #     num_elements = batch.shape[0] * batch.shape[1] * batch.shape[2] * batch.shape[3]
        #     acc += (torch.round(logits) == batch).float().sum() / num_elements
        #     train_loss += float(loss)
        #     print(f"epoch {epoch}, batch {batch_idx}, train loss {loss.data}")
        #     loss.backward()
        #     optimizer.step()
        # print(f"average train loss {epoch}: {train_loss / (inference_train_size // batch_size)}, acc {acc / (inference_train_size // batch_size)}")
        # # test_loss = 0
        # # acc = 0
        # # model.eval()
        # # for batch_idx in range(inference_test_size // batch_size):
        # #     with torch.no_grad():
        # #         batch = get_syn_batch(syn_gen_test)
        # #         logits = model(batch)
        # #         loss = model.loss_function(logits, batch)
        # #         num_elements = batch.shape[0] * batch.shape[1] * batch.shape[2] * batch.shape[3]
        # #         acc += (torch.round(logits) == batch).float().sum() / num_elements
        # #         test_loss += float(loss)
        # # print(f"average test loss {epoch}: {test_loss / (inference_test_size // batch_size)}, acc {acc / (inference_test_size // batch_size)}")
        #
        # torch.save(model.state_dict(), f"trained_models/fid-model2.pth")
    torch.save(random_images, "random_images.pt")
