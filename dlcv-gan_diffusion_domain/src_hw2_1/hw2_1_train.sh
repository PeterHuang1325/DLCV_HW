#!/bin/bash

# TODO - run your inference Python3 code
python3  -u main.py --epochs=100\
            --model_type='DCGAN'\
            --lr_D=2e-4\
            --lr_G=2e-4\
            --batch_size=64\
            --z_dim=100\
            --n_critic=1\
            --num_workers=2\
            --seed=1000 \
            --save_path='./gan_results/'\
            --workspace_dir='../hw2_data/face/'\
