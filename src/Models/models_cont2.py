"""
Defines Neural Networks
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from ..utils.generators.mixed_len_generator import Parser, \
    SimulateStack
from typing import List
from globals import device

class Encoder(nn.Module):
    def __init__(self, dropout=0.2):
        """
        Encoder for 2D CSGNet.
        :param dropout: dropout
        """
        super(Encoder, self).__init__()
        self.p = dropout
        self.conv1 = nn.Conv2d(1, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.drop = nn.Dropout(dropout)

    def encode(self, x):
        x = F.max_pool2d(self.drop(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.drop(F.relu(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ImitateJoint(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 encoder,
                 time_steps=3,
                 dropout=0.5):
        """
        Defines RNN structure that takes features encoded by CNN and produces program
        instructions at every time step.
        :param num_draws: Total number of tokens present in the dataset or total number of operations to be predicted + a stop symbol = 400
        :param canvas_shape: canvas shape
        :param dropout: dropout
        :param hd_sz: rnn hidden size
        :param input_size: input_size (CNN feature size) to rnn
        :param encoder: Feature extractor network object
        :param mode: Mode of training, RNN, BDRNN or something else
        :param num_layers: Number of layers to rnn
        :param time_steps: max length of program
        """
        super(ImitateJoint, self).__init__()
        self.hd_sz = hidden_size
        self.in_sz = input_size
        self.encoder = encoder
        self.out_sz = output_size

        # Dense layer to project input ops(labels) to input of rnn (EDITED TO EMBEDDING)
        self.emb_size = 64
        self.param_size = 64
        self.embedding = nn.Embedding(8, self.emb_size)
        self.dense_params = nn.Linear(3, self.param_size)
        self.input_op_sz = self.emb_size + self.param_size

        self.rnn = nn.GRU(
            input_size=self.in_sz + self.input_op_sz,
            hidden_size=self.hd_sz,
            batch_first=True)

        self.dense_fc_1 = nn.Linear(
            in_features=self.hd_sz, out_features=self.hd_sz)
        self.dense_output = nn.Linear(
            in_features=self.hd_sz, out_features=self.out_sz)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data, input_op, program_len):
        """
        returns (batch, timesteps, features)
        """

        assert data.size()[0] == program_len + 1, "Incorrect stack size!!"
        batch_size = data.size()[1]
        h = Variable(torch.zeros(1, batch_size, self.hd_sz)).to(device)
        x_f = self.encoder.encode(data[-1, :, 0:1, :, :])
        x_f = x_f.view(batch_size, 1, self.in_sz)

        # remove stop token for input to decoder
        input_op = input_op[:, :-1, :]

        # add some noise to params during training
        input_op[:, :, 1:3] += 8*torch.randn_like(input_op[:, :, 1:3]).to(device) # location
        input_op[:, :, 3:] += 4*torch.randn_like(input_op[:, :, 3:]).to(device) # scale
        input_params = self.dense_params(input_op[:, :, 1:])

        #input_params = torch.zeros((batch_size, input_op.shape[1], 64)).to(device)
        input_type = self.embedding(input_op[:, :, 0].long())
        input_op_rnn = torch.cat([input_type, input_params], dim=2)
        x_f = x_f.repeat(1, program_len+1, 1)
        input = torch.cat((self.drop(x_f), input_op_rnn), 2)
        output, h = self.rnn(input, h)
        output = self.relu(self.dense_fc_1(self.drop(output)))
        output = self.dense_output(self.drop(output))
        return output

    def test(self, data, input_op, program_len):
        batch_size = data.size()[1]
        h = Variable(torch.zeros(1, batch_size, self.hd_sz)).to(device)
        x_f = self.encoder.encode(data[-1, :, 0:1, :, :])
        x_f = x_f.view(batch_size, self.in_sz)

        outputs = []
        last_output = input_op[:, 0, :]
        for timestep in range(0, program_len + 1):
            # X_f is always input to the network at every time step
            # along with previous predicted label
            input_params = self.dense_params(last_output[:, 1:])
            #input_params = torch.zeros((batch_size, 64)).to(device)
            input_type = self.embedding(last_output[:, 0].long())
            # (timesteps, batch, features)
            input_op_rnn = self.relu(torch.cat([input_type, input_params], dim=1))
            input = torch.cat((self.drop(x_f), input_op_rnn), 1).reshape((batch_size, 1, -1))
            rnn_out, h = self.rnn(input, h)
            hd = self.relu(self.dense_fc_1(self.drop(rnn_out[:, 0])))
            output = self.dense_output(self.drop(hd))
            type = torch.argmax(output[:, :8], dim=1).float()
            params = F.relu(output[:, 8:])
            last_output = torch.cat([type.reshape((batch_size, 1)), params], dim=1)
            outputs.append(output)
        return torch.stack(outputs).permute(1, 0, 2)

    def loss_function(self, outputs, labels, program_len):
        # remove start token from label
        labels = labels[:, 1:, :]

        type_loss = F.cross_entropy(outputs[:, :, :8].permute(0, 2, 1), labels[:, :, 0].long())
        param_loss = F.mse_loss(outputs[:, :, 8:], labels[:, :, 1:])
        # scaling factor chosen to make param_loss and type_loss about equal
        param_loss *= 0.01
        # print(param_loss/type_loss)
        return type_loss + param_loss


class ParseModelOutput:
    def __init__(self, stack_size, canvas_shape):
        """
        This class parses complete output from the network which are in joint
        fashion. This class can be used to generate final canvas and
        expressions.
        :param unique_draws: Unique draw/op operations in the current dataset
        :param stack_size: Stack size
        :param steps: Number of steps in the program
        :param canvas_shape: Shape of the canvases
        """
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size
        self.Parser = Parser()
        self.sim = SimulateStack(self.stack_size, self.canvas_shape)

    def get_final_canvas(self,
                         outputs: List,
                         if_just_expressions=False,
                         if_pred_images=False):
        """
        Takes the raw output from the network and returns the predicted
        canvas. The steps involve parsing the outputs into expressions,
        decoding expressions, and finally producing the canvas using
        intermediate stacks.
        :param if_just_expressions: If only expression is required than we
        just return the function after calculating expressions
        :param outputs: List, each element correspond to the output from the
        network
        :return: stack: Predicted final stack for correct programs
        :return: correct_programs: Indices of correct programs
        """
        batch_size = outputs.size()[0]
        steps = outputs.size()[1]

        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        type_labels = torch.argmax(outputs[:, :, :8], dim=2).data.cpu().numpy()

        for j in range(batch_size):
            for i in range(steps):
                if type_labels[j][i] == 0:
                    expressions[j] += "+"
                if type_labels[j][i] == 1:
                    expressions[j] += "*"
                if type_labels[j][i] == 2:
                    expressions[j] += "-"
                if type_labels[j][i] == 3:
                    expressions[j] += "$"
                if type_labels[j][i] > 3:
                    params = F.relu(outputs[j, i, 8:])
                    print(params)
                    params = params.cpu().numpy().reshape((-1,))
                    params = str([int(x) for x in params])[1:-1].replace(" ", "")
                    print(params)
                    if type_labels[j][i] == 4:
                        expressions[j] += f"c({params})"
                    if type_labels[j][i] == 5:
                        expressions[j] += f"s({params})"
                    if type_labels[j][i] == 6:
                        expressions[j] += f"t({params})"

        # Remove the stop symbol and later part of the expression
        for index, exp in enumerate(expressions):
            expressions[index] = exp.split("$")[0]
        if if_just_expressions:
            return expressions
        stacks = []
        for index, exp in enumerate(expressions):
            # print(exp)
            program = self.Parser.parse(exp)
            if validity(program, len(program), len(program) - 1):
                correct_programs.append(index)
            else:
                if if_pred_images:
                    # if you just want final predicted image
                    stack = np.zeros((self.canvas_shape[0],
                                      self.canvas_shape[1]))
                else:
                    stack = np.zeros(
                        (self.steps + 1, self.stack_size, self.canvas_shape[0],
                         self.canvas_shape[1]))
                stacks.append(stack)
                continue
                # Check the validity of the expressions

            self.sim.generate_stack(program)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            if if_pred_images:
                stacks.append(stack[-1, 0, :, :])
            else:
                stacks.append(stack)
        if len(stacks) == 0:
            return None
        if if_pred_images:
            stacks = np.stack(stacks, 0).astype(dtype=np.bool)
        else:
            stacks = np.stack(stacks, 1).astype(dtype=np.bool)
        return stacks, correct_programs, expressions

    def expression2stack(self, expressions: List):
        """Assuming all the expression are correct and coming from
        groundtruth labels. Helpful in visualization of programs
        :param expressions: List, each element an expression of program
        """
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.Parser.parse(exp)
            self.sim.generate_stack(program)
            stack = self.sim.stack_t
            stack = np.stack(stack, axis=0)
            stacks.append(stack)
        stacks = np.stack(stacks, 1).astype(dtype=np.float32)
        return stacks

    def labels2exps(self, labels: np.ndarray, steps: int):
        """
        Assuming grountruth labels, we want to find expressions for them
        :param labels: Grounth labels batch_size x time_steps
        :return: expressions: Expressions corresponding to labels
        """
        if isinstance(labels, np.ndarray):
            batch_size = labels.shape[0]
        else:
            batch_size = labels.size()[0]
            labels = labels.data.cpu().numpy()
        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        for j in range(batch_size):
            for i in range(steps):
                expressions[j] += self.unique_draws[labels[j, i]]
        return expressions


def validity(program: List, max_time: int, timestep: int):
    """
    Checks the validity of the program. In short implements a pushdown automaton that accepts valid strings.
    :param program: List of dictionary containing program type and elements
    :param max_time: Max allowed length of program
    :param timestep: Current timestep of the program, or in a sense length of
    program
    # at evey index
    :return:
    """
    num_draws = 0
    num_ops = 0
    for i, p in enumerate(program):
        if p["type"] == "draw":
            # draw a shape on canvas kind of operation
            num_draws += 1
        elif p["type"] == "op":
            # +, *, - kind of operation
            num_ops += 1
        elif p["type"] == "stop":
            # Stop symbol, no need to process further
            if num_draws > ((len(program) - 1) // 2 + 1):
                return False
            if not (num_draws > num_ops):
                return False
            return (num_draws - 1) == num_ops

        if num_draws <= num_ops:
            # condition where number of operands are lesser than 2
            return False
        if num_draws > (max_time // 2 + 1):
            # condition for stack over flow
            return False
    if (max_time - 1) == timestep:
        return (num_draws - 1) == num_ops
    return True
