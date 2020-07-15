# This script only modifies the training, so that higher len programs are
# trained better.
"""
This trains network to predict stop symbol for variable length programs.
Note that there is no padding done in RNN in contrast to traditional RNN for
variable length programs. This is mainly because of computational
efficiency of forward pass, that is, each batch contains only
programs of similar length, that implies that the program of smaller lengths
are not processed by RNN for unnecessary time steps.
Losses from all batches of different time-lengths are combined to compute
gradient and updated in the network in one go. This ensures that every update to
the network has equal contribution (or weighted by the ratio of their
batch sizes) coming from programs of different lengths.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd.variable import Variable

from src.Models.loss import losses_joint
from src.Models.models_cont2 import Encoder
from src.Models.models_cont2 import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.generators.mixed_len_generator import MixedGenerateData
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer
from itertools import product

device = torch.device("cuda")

config = read_config.Config("config_synthetic.yml")

print(config.config, flush=True)

# Encoder
encoder_net = Encoder(config.encoder_drop)
encoder_net.to(device)

data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                     5: "data/synthetic/two_ops/expressions.txt",
                     7: "data/synthetic/three_ops/expressions.txt",
                     9: "data/synthetic/four_ops/expressions.txt",
                     11: "data/synthetic/five_ops/expressions.txt",
                     13: "data/synthetic/six_ops/expressions.txt"}
# first element of list is num of training examples, and second is number of
# testing examples.
proportion = config.proportion  # proportion is in percentage. vary from [1, 100].
dataset_sizes = {
    3: [30000, 50 * proportion],
    5: [110000, 500 * proportion],
    7: [170000, 500 * proportion],
    9: [270000, 500 * proportion],
    11: [370000, 1000 * proportion],
    13: [370000, 1000 * proportion]
}
dataset_sizes = {k: [x // 100 for x in v] for k, v in dataset_sizes.items()}

generator = MixedGenerateData(
    data_labels_paths=data_labels_paths,
    batch_size=config.batch_size,
    canvas_shape=config.canvas_shape)

imitate_net = ImitateJoint(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    output_size = 8+3,
    encoder=encoder_net)
imitate_net.to(device)

max_len = max(data_labels_paths.keys())

optimizer = optim.Adam(
    imitate_net.parameters(),
    weight_decay=config.weight_decay,
    lr=config.lr)

reduce_plat = LearningRate(
    optimizer,
    init_lr=config.lr,
    lr_dacay_fact=0.2,
    patience=config.patience)
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

# returns (batch, timesteps, 4) continuous encoded labels
def labels_to_cont(labels):
    s = labels.shape
    labels_cont = np.zeros((s[0], s[1], 4))

    labels_cont[labels == 396, 0] = 0
    labels_cont[labels == 397, 0] = 1
    labels_cont[labels == 398, 0] = 2
    labels_cont[labels == 399, 0] = 3
    labels_cont[labels <= 90, 0] = 4
    labels_cont[((labels > 90) & (labels <= 259)), 0] = 5
    labels_cont[((labels > 259) & (labels < 396)), 0] = 6

    for i in range(s[0]):
        for j in range(s[1]):
            if labels[i][j] < 396:
                str = generator.unique_draw[labels[i][j]]
                sep = str.split(",")
                labels_cont[i][j][1] = int(sep[0][2:])
                labels_cont[i][j][2] = int(sep[1])
                labels_cont[i][j][3] = int(sep[2][:-1])

    # start token
    labels_cont = np.pad(labels_cont, ((0, 0), (1, 0), (0, 0)))
    labels_cont[:, 0, 0] = 7

    return labels_cont

prev_test_loss = 1e20
prev_test_cd = 1e20
prev_test_iou = 0

config.epochs = 400

for epoch in range(config.epochs):
    train_loss = 0
    Accuracies = []
    imitate_net.train()
    for batch_idx in range(config.train_size //
                           (config.batch_size * config.num_traj)):
        optimizer.zero_grad()
        loss = Variable(torch.zeros(1)).to(device).data
        acc = 0
        for _ in range(config.num_traj):
            for k in data_labels_paths.keys():
                data, labels = next(train_gen_objs[k])
                labels_cont = torch.from_numpy(labels_to_cont(labels)).to(device).float()
                data = data[:, :, 0:1, :, :]
                data = Variable(torch.from_numpy(data)).to(device)
                outputs = imitate_net(data, labels_cont, k)
                loss_k = imitate_net.loss_function(outputs, labels_cont, k) / types_prog / config.num_traj
                acc += float((torch.argmax(outputs[:, :, :8], dim=2) == labels_cont[:, 1:, 0]).float().sum()) \
                       / (len(labels_cont) * (k+1)) / types_prog / config.num_traj
                loss_k.backward()
                loss += loss_k.data
                del loss_k

        #torch.nn.utils.clip_grad_norm_(imitate_net.parameters(), 1e-5)
        optimizer.step()
        train_loss += loss
        print(f"batch {batch_idx} train loss: {loss.cpu().numpy()}")
        print(f"acc: {acc}")

    mean_train_loss = train_loss / (config.train_size // (config.batch_size))
    print(f"epoch {epoch} mean train loss: {mean_train_loss.cpu().numpy()}")

    imitate_net.eval()
    loss = Variable(torch.zeros(1)).to(device)
    metrics = {"cos": 0, "iou": 0, "cd": 0}
    IOU = 0
    COS = 0
    CD = 0
    correct_programs = 0
    pred_programs = 0
    for batch_idx in range(config.test_size // (config.batch_size)):
        parser = ParseModelOutput(max_len // 2 + 1, config.canvas_shape)
        for k in data_labels_paths.keys():
            with torch.no_grad():
                data_, labels = next(test_gen_objs[k])
                labels_cont = torch.from_numpy(labels_to_cont(labels)).to(device).float()
                data = data_[:, :, 0:1, :, :]
                data = Variable(torch.from_numpy(data)).to(device)
                outputs = imitate_net.test(data, labels_cont, k)
                loss += imitate_net.loss_function(outputs, labels_cont, k) / types_prog
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    outputs, if_just_expressions=False, if_pred_images=True)
                correct_programs += len(correct_prog)
                pred_programs += len(pred_prog)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)
                iou = np.sum(np.logical_and(target_images, pred_images),
                             (1, 2)) / \
                      np.sum(np.logical_or(target_images, pred_images),
                             (1, 2))
                cos = cosine_similarity(target_images, pred_images)
                CD += np.sum(chamfer(target_images, pred_images))
                IOU += np.sum(iou)
                COS += np.sum(cos)

    metrics["iou"] = IOU / config.test_size
    metrics["cos"] = COS / config.test_size
    metrics["cd"] = CD / config.test_size

    test_losses = loss.data
    test_loss = test_losses.cpu().numpy() / (config.test_size //
                                             (config.batch_size))

    # reduce_plat.reduce_on_plateu(metrics["cd"])
    print("Epoch {}/{}=>  train_loss: {}, iou: {}, cd: {}, test_mse: {}".format(epoch, config.epochs,
                                      mean_train_loss.cpu().numpy(),
                                      metrics["iou"], metrics["cd"], test_loss,))
    print(f"CORRECT PROGRAMS: {correct_programs}")
    print(f"PREDICTED PROGRAMS: {pred_programs}")
    print(f"RATIO: {correct_programs/pred_programs}")

    del test_losses, outputs
    if prev_test_cd > metrics["cd"]:
        print("Saving the Model weights based on CD", flush=True)
        torch.save(imitate_net.state_dict(),
                   "trained_models/synthetic_cont.pth")
        prev_test_cd = metrics["cd"]
