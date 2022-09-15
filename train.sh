#!/usr/bin/env bash

seed="42"
gpu='1'

config='spineweb_ours'
default_command="--seed ${seed} --config ${config}"
custom_command=""
CUDA_VISIBLE_DEVICES="${gpu}" python -u main.py ${default_command} ${custom_command} --save_test_prediction
