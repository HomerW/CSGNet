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
from src.Models.models_perturb import Encoder
from src.Models.models_perturb import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.generators.mixed_len_generator_perturb import MixedGenerateData
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer
import torch.nn.functional as F
from globals import device

config = read_config.Config("config_synthetic.yml")

print(config.config, flush=True)

# Encoder
encoder_net = Encoder(config.encoder_drop)
encoder_net.cuda()

# data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
#                      5: "data/synthetic/two_ops/expressions.txt",
#                      7: "data/synthetic/three_ops/expressions.txt",
#                      9: "data/synthetic/four_ops/expressions.txt",
#                      11: "data/synthetic/five_ops/expressions.txt",
#                      13: "data/synthetic/six_ops/expressions.txt"}
# # first element of list is num of training examples, and second is number of
# # testing examples.
# proportion = config.proportion  # proportion is in percentage. vary from [1, 100].
# dataset_sizes = {
#     3: [25000, 50 * proportion],
#     5: [100000, 500 * proportion],
#     7: [150000, 500 * proportion],
#     9: [250000, 500 * proportion],
#     11: [350000, 1000 * proportion],
#     13: [350000, 1000 * proportion]
# }
# dataset_sizes = {k: [x // 1000 for x in v] for k, v in dataset_sizes.items()}
data_labels_paths = {3: "data/synthetic/one_op/expressions.txt",
                     5: "data/synthetic/two_ops/expressions.txt",
                     7: "data/synthetic/three_ops/expressions.txt"}
# first element of list is num of training examples, and second is number of
# testing examples.
proportion = config.proportion  # proportion is in percentage. vary from [1, 100].
dataset_sizes = {
    3: [proportion * 250, proportion * 50],
    5: [proportion * 1000, proportion * 100],
    7: [proportion * 1500, proportion * 200]
}
dataset_sizes = {k: [x // 100 for x in v] for k, v in dataset_sizes.items()}

generator = MixedGenerateData(
    data_labels_paths=data_labels_paths,
    batch_size=config.batch_size,
    canvas_shape=config.canvas_shape)

imitate_net = ImitateJoint(
    hd_sz=config.hidden_size,
    input_size=config.input_size,
    encoder=encoder_net,
    mode=config.mode,
    num_draws=len(generator.unique_draw),
    canvas_shape=config.canvas_shape,
    teacher_force=True)
imitate_net.cuda()

# if config.preload_model:
#     print("pre loading model")
#     pretrained_dict = torch.load("trained_models/small_test_perturb.pth")
#     imitate_net_dict = imitate_net.state_dict()
#     pretrained_dict = {
#         k: v
#         for k, v in pretrained_dict.items() if k in imitate_net_dict
#     }
#     imitate_net_dict.update(pretrained_dict)
#     imitate_net.load_state_dict(imitate_net_dict)

for param in imitate_net.parameters():
    param.requires_grad = True

for param in encoder_net.parameters():
    param.requires_grad = True

max_len = max(dataset_sizes.keys())

optimizer = optim.Adam(
    [para for para in imitate_net.parameters() if para.requires_grad],
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
for k in dataset_sizes.keys():
    test_batch_size = int(config.batch_size * dataset_sizes[k][1] / \
                          config.test_size)
    # Acts as a curriculum learning
    train_batch_size = config.batch_size // types_prog
    train_gen_objs[k] = generator.get_train_data(
        train_batch_size,
        k,
        num_train_images=dataset_sizes[k][0],
        jitter_program=False)
    test_gen_objs[k] = generator.get_test_data(
        test_batch_size,
        k,
        num_train_images=dataset_sizes[k][0],
        num_test_images=dataset_sizes[k][1],
        jitter_program=False)

prev_test_loss = 1e20
prev_test_cd = 1e20
prev_test_iou = 0
for epoch in range(config.epochs):
    train_loss = 0
    Accuracies = []
    imitate_net.train()
    for batch_idx in range(config.train_size //
                           (config.batch_size * config.num_traj)):
        optimizer.zero_grad()
        loss = Variable(torch.zeros(1)).cuda().data
        loss_p = Variable(torch.zeros(1)).cuda().data
        loss_t = Variable(torch.zeros(1)).cuda().data
        acc = 0
        for _ in range(config.num_traj):
            for k in dataset_sizes.keys():
                data, labels, perturbs = next(train_gen_objs[k])
                data = data[:, :, 0:1, :, :]
                one_hot_labels = prepare_input_op(labels,
                                                  len(generator.unique_draw))
                one_hot_labels = Variable(
                    torch.from_numpy(one_hot_labels)).cuda()
                data = Variable(torch.from_numpy(data)).cuda()
                labels = Variable(torch.from_numpy(labels)).cuda()
                outputs, perturb_out = imitate_net([data, one_hot_labels, k])
                perturbs = torch.from_numpy(perturbs).to(device)
                perturb_out = perturb_out.permute(1, 0, 2)

                # mask off ops and stop token
                perturb_loss = F.mse_loss(perturbs[labels < 396], perturb_out[labels < 396]) / len(dataset_sizes.keys()) / config.num_traj
                #perturb_loss = F.mse_loss(perturbs, perturb_out) / len(dataset_sizes.keys()) / config.num_traj
                if not imitate_net.tf:
                    acc += float((torch.argmax(torch.stack(outputs), dim=2).permute(1, 0) == labels).float().sum()) \
                           / (labels.shape[0] * labels.shape[1]) / types_prog / config.num_traj
                else:
                    acc += float((torch.argmax(outputs, dim=2).permute(1, 0) == labels).float().sum()) \
                           / (labels.shape[0] * labels.shape[1]) / types_prog / config.num_traj
                loss_k_token = ((losses_joint(outputs, labels, time_steps=k + 1) / (
                    k + 1)) / len(dataset_sizes.keys()) / config.num_traj)
                #loss_k = loss_k_token + perturb_loss
                loss_k = loss_k_token
                loss_k.backward()
                loss += loss_k.data
                loss_p += perturb_loss.data
                loss_t += loss_k_token.data
                del loss_k

        optimizer.step()
        train_loss += loss
        print(f"batch {batch_idx} train loss: {loss.cpu().numpy()}, token loss: {loss_t.cpu().numpy()}, perturb loss: {loss_p.cpu().numpy()}")
        print(f"acc: {acc}")

    mean_train_loss = train_loss / (config.train_size // (config.batch_size))
    print(f"epoch {epoch} mean train loss: {mean_train_loss.cpu().numpy()}")
    imitate_net.eval()
    loss = Variable(torch.zeros(1)).cuda()
    loss_p = Variable(torch.zeros(1)).cuda().data
    loss_t = Variable(torch.zeros(1)).cuda().data
    acc = 0
    metrics = {"cos": 0, "iou": 0, "cd": 0}
    IOU = 0
    COS = 0
    CD = 0
    correct_programs = 0
    pred_programs = 0
    for batch_idx in range(config.test_size // (config.batch_size)):
        parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len,
                          config.canvas_shape)
        for k in dataset_sizes.keys():
            with torch.no_grad():
                data_, labels, perturbs = next(test_gen_objs[k])
                one_hot_labels = prepare_input_op(labels, len(
                    generator.unique_draw))
                one_hot_labels = Variable(torch.from_numpy(one_hot_labels)).cuda()
                data = Variable(torch.from_numpy(data_)).cuda()
                labels = Variable(torch.from_numpy(labels)).cuda()
                test_outputs, perturb_outputs = imitate_net([data, one_hot_labels, k])
                loss_token = (losses_joint(test_outputs, labels, time_steps=k + 1) /
                         (k + 1)) / types_prog
                perturbs = torch.from_numpy(perturbs).to(device)
                perturb_outputs = perturb_outputs.permute(1, 0, 2)
                perturb_loss = F.mse_loss(perturbs[labels < 396], perturb_outputs[labels < 396]) / len(dataset_sizes.keys())
                # perturb_loss = F.mse_loss(perturbs, perturb_outputs) / len(dataset_sizes.keys())
                #loss += loss_token + perturb_loss
                loss += loss_token
                loss_t += loss_token
                loss_p += perturb_loss
                test_output, perturb_out = imitate_net.test([data, one_hot_labels, max_len])
                acc += float((torch.argmax(torch.stack(test_output), dim=2)[:k].permute(1, 0) == labels[:, :-1]).float().sum()) \
                        / (len(labels) * (k+1)) / types_prog / (config.test_size // config.batch_size)
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_output, perturb_out, if_just_expressions=False, if_pred_images=True)
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
    test_loss_t = loss_t.data.cpu().numpy() / (config.test_size //
                                             (config.batch_size))
    test_loss_p = loss_p.data.cpu().numpy() / (config.test_size //
                                             (config.batch_size))

    reduce_plat.reduce_on_plateu(metrics["cd"])
    print("Epoch {}/{}=>  train_loss: {}, iou: {}, cd: {}, test_mse: {}, test_acc: {}, token_loss: {}, perturb_loss: {}".format(epoch, config.epochs,
                                      mean_train_loss.cpu().numpy(),
                                      metrics["iou"], metrics["cd"], test_loss, acc, test_loss_t, test_loss_p))
    print(f"CORRECT PROGRAMS: {correct_programs}")
    print(f"PREDICTED PROGRAMS: {pred_programs}")
    print(f"RATIO: {correct_programs/pred_programs}")

    del test_losses, test_outputs
    # if prev_test_cd > metrics["cd"]:
    #     print("Saving the Model weights based on CD", flush=True)
    #     torch.save(imitate_net.state_dict(),
    #                "trained_models/small_test.pth")
    #     prev_test_cd = metrics["cd"]
