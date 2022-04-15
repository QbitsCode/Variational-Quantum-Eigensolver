#!/bin/bash

FILENAME=pec_fs_cmd.py
# FILENAME=pec_ground_cmd.py
# FILENAME=sp_cmd.py
# FILENAME=pec_eigen_cmd.py

JOBPID=$$
echo $JOBPID

python3 $FILENAME \
        --jobID=$JOBPID  \
        --algo='VQE_fs' \
        --molecule='LiH' \
        --nlayer=1 \
        --omega=-7.5 \
        --basis='small_custom' \
        --bondlen=0 \
        --refstate='HF' \
        --device='CPU' \
        --lb=0.7 \
        --ub=2.1 \
        --step=0.1 \



