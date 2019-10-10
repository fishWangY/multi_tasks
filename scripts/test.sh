#!/usr/bin/env bash
source activate classify
TEST_DATA_DIR='./'
DATA_DIR="./"
MODEL_DIR="./models/"
MODEL_NAME="inception_v4"
EVAL_INTERVAL=200

cd ..
cd /src
while true
do
    python -u test.py

    sleep ${EVAL_INTERVAL}

done