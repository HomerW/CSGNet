"""Defines the configuration to be loaded before running any experiment"""

#from configobj import ConfigObj
import string


class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """

        self.filename = filename
        config = {'train':{}}
        with open(self.filename) as f:
            for line in f:
                ls = line.split()
                if len(ls) == 0:
                    continue
                if ls[0][0] == '#':
                    continue
                if ls[0][0] == '[':
                    continue

                key = line.split()[0]
                value = line.split()[2]
                config['train'][key] = value
                
                    
        self.config = config

        # Comments on the experiments running
        self.comment = 'blah'

        # Model name and location to store
        self.model_path = config["train"]["model_path"]

        # Whether to load a pretrained model or not
        self.preload_model = bool(config["train"]["preload_model"])

        # path to the model
        self.pretrain_modelpath = config["train"]["pretrain_model_path"]

        # Number of batches to be collected before the network update
        self.num_traj = int(config["train"]["num_traj"])

        # Number of epochs to run during training
        self.epochs = int(config["train"]["num_epochs"])

        # batch size, based on the GPU memory
        self.batch_size = int(config["train"]["batch_size"])

        # hidden size of RNN
        self.hidden_size = int(config["train"]["hidden_size"])

        # Output feature size from CNN
        self.input_size = int(config["train"]["input_size"])

        # Mode of training, 1: supervised, 2: RL
        self.mode = int(config["train"]["mode"])

        # Learning rate
        self.lr = float(config["train"]["lr"])

        # Encoder drop
        self.encoder_drop = float(config["train"]["encoder_drop"])

        # l2 Weight decay
        self.weight_decay = float(config["train"]["weight_decay"])

        # dropout for Decoder network
        self.dropout = float(config["train"]["dropout"])

        # Number of epochs to wait before decaying the learning rate.
        self.patience = int(config["train"]["patience"])

        # Optimizer: RL training -> "sgd" or supervised training -> "adam"
        self.optim = config["train"]["optim"]

        # Proportion of the dataset to be used while training, use 100
        self.proportion = int(config["train"]["proportion"])

        # Epsilon for the RL training, not applicable in Supervised training
        self.eps = float(config["train"]["epsilon"])

        # Whether to schedule the learning rate or not
        self.lr_sch = bool(config["train"]["lr_sch"])

        # Canvas shape, keep it [64, 64]
        self.canvas_shape = [int(config["train"]["canvas_shape"])] * 2

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and
        what parameters have been used.
        :return:
        """
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config_synthetic.yml")
    print(file.write_config())
