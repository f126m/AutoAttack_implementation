#!/bin/sh

CUDA_VISIBLE_DEVICES=7 python main.py --data_dir /datasets/MNIST --model ./trained_models/mnist-Linf-MMA-0.45-sd0/model_best.pt