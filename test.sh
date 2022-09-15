#!/usr/bin/env bash

seed="42"
gpu='1'

config='_'
default_command="--seed ${seed} --config ${config}"
custom_command=""
CUDA_VISIBLE_DEVICES="${gpu}" python -u main.py ${default_command} ${custom_command} --save_test_prediction --only_test_version "ExpNum[00001]_Dataset[dataset16]_Model[RITM_SE_HRNet32]_config[spineweb_ours]_seed[42]"
