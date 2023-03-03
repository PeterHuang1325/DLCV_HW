#!/bin/bash


echo 'The arguments passed in are:' $@

#cd './src_hw1_1'

# TODO - run your inference Python3 code
python3  -u src_hw1_1/inference.py --input_path=$1 \
                    --output_path=$2
