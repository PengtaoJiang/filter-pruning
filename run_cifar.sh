#!/bin/bash

# arg1: GPU ID
# arg2: prune type
# arg3: prune-rate

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "arg1: GPU ID, arg2: network depth, arg3: prune-rate"
    exit
fi

MODEL="cifar_resnet"$2"_v1"
TMP="tmp/cifar-prune-type/prune-type-"$2

echo "Model name: "$MODEL
echo "tmp folder: "$TMP

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
	CUDA_VISIBLE_DEVICES=$1 python end2end.py --prune-type $2 --tmp $TMP
done