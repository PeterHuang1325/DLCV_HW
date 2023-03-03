#!/bin/bash

# TODO - run your inference Python3 code
echo 'The arguments passed in are:' $@

# TODO - run your inference Python3 code
python3  -u byol-pytorch/fine_tune/inference.py --csv_path=$1 \
                    --input_path=$2 \
                    --output_path=$3 \
                    