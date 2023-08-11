#!/usr/bin/env sh

export MXNET_ENGINE_TYPE=NaiveEngine
python -u detection.py --img mxnet-face-fr50-roc.png --gpu 0 
