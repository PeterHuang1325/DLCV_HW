#!/bin/bash

# TODO - run your inference Python3 code
echo 'The arguments passed in are:' $@

#cd './src_hw1_1'

# TODO - run your inference Python3 code
python3  -u src_hw3_1/inference.py --data_path=$1 \
                    --json_path=$2 \
                    --output_path=$3  