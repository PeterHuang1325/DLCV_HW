#!/bin/bash

# TODO - run your inference Python3 code

echo 'The arguments passed in are:' $@

cd './src_hw2_2'

# TODO - run your inference Python3 code
python3  -u sample_images.py --output_path=$1 \
