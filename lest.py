import torch
from src.Models.models import Encoder
from src.Models.models import ImitateJoint, ParseModelOutput
from src.utils import read_config
from src.utils.train_utils import image_from_expressions
from infer import infer_programs
from train import train
import time

device = torch.device("cuda")

"""
Get initial pretrained CSGNet inference network
"""
def get_csgnet():
    config = read_config.Config("config_synthetic.yml")

    # Encoder
    encoder_net = Encoder(config.encoder_drop)
    encoder_net = encoder_net.to(device)

    imitate_net = ImitateJoint(
        hd_sz=config.hidden_size,
        input_size=config.input_size,
        encoder=encoder_net,
        mode=config.mode,
        num_draws=400,
        canvas_shape=config.canvas_shape)
    imitate_net = imitate_net.to(device)

    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath, map_location=device)
    imitate_net_dict = imitate_net.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_dict)
    imitate_net.load_state_dict(imitate_net_dict)

    return imitate_net

"""
Run LEST
"""
def lest(iterations):
    csgnet = get_csgnet()

    inf_epochs = 0

    for i in range(iterations):
        print(f"ROUND {i}")

        infer_path = "train_out/"

        infer_programs(csgnet, infer_path)
        inf_epochs += train(csgnet, infer_path)

        torch.save(csgnet.state_dict(), f"{infer_path}/model.pt")

        print(f"Total inference epochs: {inf_epochs}")

if __name__ == '__main__':
    lest(200)
