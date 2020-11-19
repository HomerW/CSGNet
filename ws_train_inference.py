import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, ParseModelOutput
from src.Models.loss import losses_joint
from src.utils import read_config
from src.utils.generators.wake_sleep_gen import WakeSleepGen
from src.utils.learn_utils import LearningRate
from src.utils.train_utils import prepare_input_op, cosine_similarity, chamfer, beams_parser, validity, image_from_expressions, stack_from_expressions
from globals import device
import time
from src.utils.generators.shapenet_generater import Generator

max_len = 13
beam_width = 10
THRESH = .001
PAT = 5
LR = 0.001
EVAL_PER = 5

def train_inference(imitate_net, path, num_train, num_test, batch_size, self_training, round, mode):

    epochs = 400
    config = read_config.Config("config_synthetic.yml")
    config.batch_size = batch_size
    config.lr = LR
    patience = PAT
    
    train_size = num_train

    generator = WakeSleepGen(
        f"{path}/",
        config.batch_size,
        train_size,
        config.canvas_shape,
        max_len,
        self_training,
        round,
        mode        
    )
        
    train_gen = generator.get_train_data()
    
    cad_generator = Generator()

    val_gen = cad_generator.val_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    test_gen = cad_generator.test_gen(
        batch_size=config.batch_size,
        path="data/cad/cad.h5",
        if_augment=False)

    # FREEZE!
    for parameter in imitate_net.encoder.parameters():
        parameter.requires_grad = False
    
    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        lr=config.lr)

    # todo -> had a reduce plat here
    
    best_dict_path = f'{path}/best_dict.pt'

    torch.save(imitate_net.state_dict(), best_dict_path)
    
    best_val_cd = 1e20
    
    num_worse = 0

    for epoch in range(epochs):
        
        start = time.time()
        train_loss = 0
        
        imitate_net.train()
        for batch_idx in range(train_size // config.batch_size):            
            optimizer.zero_grad()
            loss = 0
            data, labels = next(train_gen)
            one_hot_labels = prepare_input_op(labels,
                                                  len(generator.unique_draw))
            one_hot_labels = torch.from_numpy(one_hot_labels).to(device)
            data = data.to(device)
            labels = labels.to(device)
            outputs = imitate_net([data, one_hot_labels, max_len])
            loss_k = (losses_joint(outputs, labels, time_steps=max_len + 1) / (
                max_len + 1))
            loss_k.backward()
            loss += float(loss_k)
            del loss_k
            optimizer.step()
            train_loss += loss

        mean_train_loss = train_loss / (train_size // (config.batch_size))
        
        if (epoch + 1) % EVAL_PER != 0:
            print(f"Epoch {epoch}/{epochs} =>  train_loss: {mean_train_loss}")
            continue

        imitate_net.eval()
        
        metrics = {"cd": 0}
        
        CD = 0
        for batch_idx in range(num_test // config.batch_size):
            parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
            with torch.no_grad():
                labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
                data_ = next(val_gen)
                one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).cuda()
                data = torch.from_numpy(data_).cuda()
                test_outputs = imitate_net.test([data[-1, :, 0, :, :], one_hot_labels, max_len])
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_outputs, if_just_expressions=False, if_pred_images=True)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)

                CD += np.sum(chamfer(target_images, pred_images))
                
        metrics["cd"] = CD / num_test

        test_losses = loss
        test_loss = test_losses / (num_test //
                                   (config.batch_size))

        if metrics["cd"] >= best_val_cd - THRESH:
            num_worse += 1
        else:
            num_worse = 0
            best_val_cd = metrics["cd"]
            torch.save(imitate_net.state_dict(), best_dict_path)
            
        if num_worse >= patience:
            imitate_net.load_state_dict(torch.load(best_dict_path))
            return epoch + 1

        test_metrics = {"cd": 0}

        CD = 0
        for batch_idx in range(num_test // config.batch_size):
            parser = ParseModelOutput(generator.unique_draw, max_len // 2 + 1, max_len,
                              config.canvas_shape)
            with torch.no_grad():
                labels = np.zeros((config.batch_size, max_len), dtype=np.int32)
                data_ = next(test_gen)
                one_hot_labels = prepare_input_op(labels, len(generator.unique_draw))
                one_hot_labels = torch.from_numpy(one_hot_labels).cuda()
                data = torch.from_numpy(data_).cuda()
                test_outputs = imitate_net.test([data[-1, :, 0, :, :], one_hot_labels, max_len])
                pred_images, correct_prog, pred_prog = parser.get_final_canvas(
                    test_outputs, if_just_expressions=False, if_pred_images=True)
                target_images = data_[-1, :, 0, :, :].astype(dtype=bool)

                CD += np.sum(chamfer(target_images, pred_images))
                
        test_metrics["cd"] = CD / num_test
        end = time.time()
        print(f"Epoch {epoch}/{epochs} =>  train_loss: {mean_train_loss}, val cd: {metrics['cd']}, test cd {test_metrics['cd']} | time {end-start}")        

        del test_losses, outputs, test_outputs
        
    return epochs
