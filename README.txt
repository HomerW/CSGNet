This code runs LEST using code adapted from
2D CSGNet (https://github.com/Hippogriff/CSGNet).

The dependencies can be installed with conda.
conda env create -f environment.yml

The pretrained model provided in the CSGNet repository should be placed in
"trained_models/sp.pt." The CAD data from the same repository should be placed
in the "data" directory.

lest.py starts a run of LEST, saving models and intermediate training ouput to
the directory "train_out".

infer.py and train.py contain the logic for inferring programs using the inference
network and training the inference network.

rl.py runs the reinforcement learning comparison method.

nn.py contains code for computing bidirectional average nearest neighbor distance
using Chamfer Distance.

config_synthetic.yml and config_cad.yml contain hyperparameters for LEST and RL
respectively.



This code is adapted from the CSGNet repository which requires inclusion of
the following license:

MIT License

Copyright (c) 2018 Gopal Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
