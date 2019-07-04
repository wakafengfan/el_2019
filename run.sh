#!/bin/sh

cd el_2019
export PYTHONPATH="."
export ROOT_DIR="root"
/root/anaconda3/envs/py36/bin/python prtrain_LM/process_pretrain_data.py