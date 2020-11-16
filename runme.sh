#!/bin/bash
#
#  Execute from the current working directory
# (note #$ is a gridEngine command, rather than a bash comment, the line below is executed by gridEngine)
#$ -cwd
#
#  This is a long-running job
#$ -l inf
#
# use 1 gpu
#$ -l gpus=1
#
#  Can use up to 6GB of memory
#$ -l vf=8G
#

# Using conda to activate a virtual envrionment
# Note that you must use the absolute path to the 'activate' script. You can't
#    just run 'source activate', because the necessarily bits of .bash_profile that
#    put 'activate' in your path don't run when a grid session starts up.
virtual_env='csgnet_venv'
source /data/drlab/hwalke/csgnet_venv/bin/activate
echo $virtual_env activated

# Run whatever training script you have
# You don't need to prepend 'CUDA_VISIBLE_DEVICES' here, because GridEngine will ensure
#    that your script runs in an environment with only a certain number of GPUs visible
#    (Specifically, the number you requested with '-l gpus')
python -u wake-sleep.py > ws_frozen_lest_to_st.out
