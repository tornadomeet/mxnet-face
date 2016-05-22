#!/usr/bin/env sh 

align_data_path=/home/work/data/Face/LFW/lfw-align
model_prefix=../model/lightened_cnn/lightened_cnn
epoch=166
# evaluate on lfw
python lfw.py --lfw-align $align_data_path --model-prefix $model_prefix --epoch $epoch 
