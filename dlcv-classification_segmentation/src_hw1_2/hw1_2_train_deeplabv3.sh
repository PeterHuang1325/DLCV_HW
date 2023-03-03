#!/bin/bash

# TODO - run your inference Python3 code
python3  -u main.py --epochs=60\
            --model='DeeplabV3'\
            --learning_rate=5e-4\
            --image_size=512\
            --batch_size=12\
            --num_classes=7\
            --num_workers=0\
            --seed=1000 \
            --patience=15\
