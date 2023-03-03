#!/bin/bash

# TODO - run your inference Python3 code

echo 'The arguments passed in are:' $@

cd './src_hw2_1'

# TODO - run your inference Python3 code
python3  -u inference.py --output_path=$1 \
