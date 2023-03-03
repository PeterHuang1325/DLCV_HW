#!/bin/bash

# TODO - run your inference Python3 code

echo 'The arguments passed in are:' $@

#cd './src_hw2_3'

# TODO - run your inference Python3 code
python3  -u src_hw2_3/inference.py --input_path=$1 \
                        --output_path=$2
