#!/bin/bash

# TODO - run your inference Python3 code
python3  -u main.py --epochs=50\
            --learning_rate=5e-4\
            --image_size=32\
            --batch_size=250\
            --num_classes=50\
            --num_workers=2\
            --seed=0 \
            --patience=10\
