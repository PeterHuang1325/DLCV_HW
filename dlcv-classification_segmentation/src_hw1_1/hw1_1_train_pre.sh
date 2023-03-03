#!/bin/bash

# TODO - run your inference Python3 code
python3  -u main.py --epochs=50\
            --pretrained\
            --learning_rate=5e-3\
            --image_size=128\
            --batch_size=125\
            --num_classes=50\
            --num_workers=0\
            --seed=0\
            --patience=10\