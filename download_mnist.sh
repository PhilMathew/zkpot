#! /usr/bin/env bash

mkdir ./data/
wget -O ./data/test.parquet https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet\?download\=true
wget -O ./data/train.parquet https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet\?download\=true
echo "MNIST parquet files downloaded to $pwd/data"
