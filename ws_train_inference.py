import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from src.Models.models_perturb import Encoder
from src.Models.models_perturb import ImitateJoint, ParseModelOutput
from src.Models.loss import losses_joint
from src.utils import read_config
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer, beams_parser, validity, image_from_expressions, stack_from_expressions
from globals import device

inference_train_size = 10000
inference_test_size = 3000
vocab_size = 400
generator_hidden_dim = 256
generator_latent_dim = 20
max_len = 13

"""
Trains CSGNet to convergence on samples from generator network
TODO: train to convergence and not number of epochs
"""
def train_inference(imitate_net, path):
    config = read_config.Config("config_synthetic.yml")

    generator = WakeSleepGen(f"{path}/labels.pt",
                             f"{path}/val/labels.pt",
                             batch_size=config.batch_size,
                             train_size=inference_train_size,
                             test_size=inference_test_size,
                             canvas_shape=config.canvas_shape,
                             max_len=max_len)

    train_gen = generator.get_train_data()
    test_gen = generator.get_test_data()

    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        lr=config.lr)

    reduce_plat = LearningRate(
        optimizer,
        init_lr=config.lr,
        lr_dacay_fact=0.2,
        patience=config.patience)

    best_test_loss = 1e20
    best_imitate_dict = imitate_net.state_dict()

    prev_test_cd = 1e20
    prev_test_iou = 0

    patience = 5
    num_worse = 0

    for epoch in range(50):
        train_loss = 0
        Accuracies = []
        imitate_net.train()
        for batch_idx in range(inference_train_size //
                               (config.batch_size * config.num_traj)):
            optimizer.zero_grad()
            loss = 0
            loss_p = 0
            loss_t = 0
            acc = 0
            for _ in range(config.num_traj):
                data, labels, perturbs = next(train_gen)
                data = data[:, :, 0:1, :, :]
                one_hot_labels = prepare_input_op(labels,
                                                  len(generator.unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = data.to(device)
                labels = labels.to(device)
                outputs, perturb_out = imitate_net([data, one_hot_labels, max_len])
                perturbs = perturbs.to(device)
                perturb_out = perturb_out.permute(1, 0, 2)
                perturb_loss = F.mse_loss(perturbs, perturb_out) / config.num_traj
                if not imitate_net.tf:
                    acc += float((torch.argmax(torch.stack(outputs), dim=2).permute(1, 0) == labels).float().sum()) \
                           / (labels.shape[0] * labels.shape[1]) / config.num_traj
                else:
                    acc += float((torch.argmax(outputs, dim=2).permute(1, 0) == labels).float().sum()) \
                           / (labels.shape[0] * labels.shape[1]) / config.num_traj
                loss_k_token = ((losses_joint(outputs, labels, time_steps=max_len + 1) / (
                    max_len + 1)) / config.num_traj)
                loss_k = loss_k_token + perturb_loss
                # loss_k = loss_k_token
                loss_k.backward()
                loss += float(loss_k)
                loss_p += float(perturb_loss)
                loss_t += float(loss_k_token)
                del loss_k

            optimizer.step()
            train_loss += loss
            print(f"batch {batch_idx} train loss: {loss}, token loss: {loss_t}, perturb loss: {loss_p}")
            print(f"acc: {acc}")

        mean_train_loss = train_loss / (inference_train_size // (config.batch_size))
        print(f"epoch {epoch} mean train loss: {mean_train_loss}")
        imitate_net.eval()
        loss = 0
        loss_p = 0
        loss_t = 0
        acc = 0
        metrics = {"cos": 0, "iou": 0, "cd": 0}
        IOU = 0
        COS = 0
        CD = 0
        correct_programs = 0
        pred_programs = 0
        for batch_idx in range(inference_test_size // config.batch_size):
            parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
            with torch.no_grad():
                data_, labels, perturbs = next(test_gen)
                one_hot_labels = prepare_input_op(labels, len(
                    generator.unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
                data = data_.to(device)
                labels = labels.to(device)
                test_outputs, perturb_outputs = imitate_net([data, one_hot_labels, max_len])
                loss_token = (losses_joint(test_outputs, labels, time_steps=max_len + 1) /
                         (max_len + 1))
                perturbs = perturbs.to(device)
                perturb_outputs = perturb_outputs.permute(1, 0, 2)
                perturb_loss = F.mse_loss(perturbs, perturb_outputs)
                loss += float(loss_token + perturb_loss)
                loss_t += float(loss_token)
                loss_p += float(perturb_loss)
                test_output, perturb_out = imitate_net.test([data, one_hot_labels, max_len])
                acc += float((torch.argmax(torch.stack(test_output), dim=2).permute(1, 0) == labels[:, :-1]).float().sum()) \
                        / (len(labels) * (max_len+1)) / (inference_test_size // config.batch_size)
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_output, perturb_out, if_just_expressions=False, if_pred_images=True)
                correct_programs += len(correct_prog)
                pred_programs += len(pred_prog)
                target_images = data_[-1, :, 0, :, :].cpu().numpy().astype(dtype=bool)
                iou = np.sum(np.logical_and(target_images, pred_images),
                             (1, 2)) / \
                      np.sum(np.logical_or(target_images, pred_images),
                             (1, 2))
                cos = cosine_similarity(target_images, pred_images)
                CD += np.sum(chamfer(target_images, pred_images))
                IOU += np.sum(iou)
                COS += np.sum(cos)

        metrics["iou"] = IOU / inference_test_size
        metrics["cos"] = COS / inference_test_size
        metrics["cd"] = CD / inference_test_size

        test_losses = loss
        test_loss = test_losses / (inference_test_size //
                                                 (config.batch_size))
        test_loss_t = loss_t / (inference_test_size //
                                                 (config.batch_size))
        test_loss_p = loss_p / (inference_test_size //
                                                 (config.batch_size))

        if test_loss >= best_test_loss:
            num_worse += 1
        else:
            num_worse = 0
            best_test_loss = test_loss
            best_imitate_dict = imitate_net.state_dict()
        if num_worse >= patience:
            # load the best model and stop training
            imitate_net.load_state_dict(best_imitate_dict)
            break

        reduce_plat.reduce_on_plateu(metrics["cd"])
        print("Epoch {}/{}=>  train_loss: {}, iou: {}, cd: {}, test_mse: {}, test_acc: {}, token_loss: {}, perturb_loss: {}".format(epoch, config.epochs,
                                          mean_train_loss,
                                          metrics["iou"], metrics["cd"], test_loss, acc, test_loss_p, test_loss_t))
        print(f"CORRECT PROGRAMS: {correct_programs}")
        print(f"PREDICTED PROGRAMS: {pred_programs}")
        print(f"RATIO: {correct_programs/pred_programs}")

        del test_losses, test_outputs, perturb_outputs, test_output, perturb_out
