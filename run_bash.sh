#!/bin/bash

# FILENAME=pec_fs_cmd.py
FILENAME=pec_ground_cmd.py
# FILENAME=sp_cmd.py

JOBPID=$$
echo $JOBPID

python3 $FILENAME \
        --jobID=$JOBPID  \
        --algo='VQE_g' \
        --molecule='H2' \
        --nlayer=1 \
        --omega=0 \
        --basis='small_custom' \
        --bondlen=0 \
        --refstate='HF' \
        --device='CPU' \
        --lb=0.5 \
        --ub=2 \
        --step=0.5 \



