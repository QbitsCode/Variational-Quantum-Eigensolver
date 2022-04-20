#!/bin/bash

FILENAME=pec_fs_cmd.py
# FILENAME=pec_ground_cmd.py
# FILENAME=sp_cmd.py
# FILENAME=pec_eigen_cmd.py

JOBPID=$$
echo $JOBPID

python3 $FILENAME \
        --outdir='./outputs/' \
        --jobID=$JOBPID  \
        --algo='VQE_fs' \
        --molecule='LiH' \
        --nlayer=1 \
        --omega=0 \
        --basis='small_custom' \
        --bondlen=0 \
        --refstate='HF' \
        --device='CPU' \
        --lb=1.5 \
        --ub=1.6 \
        --step=0.1 \



