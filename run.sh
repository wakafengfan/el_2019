#!/bin/sh

cd el_2019
export PYTHONPATH="."
export ROOT_DIR="root"
#/root/anaconda3/envs/py36/bin/python pretrain_LM/finetune_on_pregenerated.py
/root/anaconda3/envs/py36/bin/python baseline_4/el_pt_object.py
#/root/anaconda3/envs/py36/bin/python baseline_4/el_pt_subject.py
#/root/anaconda3/envs/py36/bin/python baseline_3/el_pt_object_submit.py