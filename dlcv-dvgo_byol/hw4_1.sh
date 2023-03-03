#!/bin/bash

# TODO - run your inference Python3 code
echo 'The arguments passed in are:' $@

# TODO - run your inference Python3 code
python3  -u DVGO/run.py --json_path=$1 \
                    --output_path=$2 