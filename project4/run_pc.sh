#!/bin/bash

# Get the current directory
CURRENT_DIR=$(pwd)
echo "Current directory: ${CURRENT_DIR}"

# Set file paths
TRAIN_X=./dataset/training/train-images.idx3-ubyte
TRAIN_Y=./dataset/training/train-labels.idx1-ubyte
TEST_X=./dataset/testing/t10k-images.idx3-ubyte
TEST_Y=./dataset/testing/t10k-labels.idx1-ubyte

# Execute the program and redirect output to PC.txt
${CURRENT_DIR}/build/softmax $TRAIN_X $TRAIN_Y $TEST_X $TEST_Y > PC.txt 2>&1

# Optionally, display a message indicating completion
echo "Execution completed. Output saved to PC.txt"
