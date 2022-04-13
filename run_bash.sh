#!/bin/bash

FILENAME=pec_fs_cmd.py
# FILENAME=pec_ground_cmd.py
# FILENAME=sp_cmd.py

JOBPID=$$
echo $JOBPID

python3 $FILENAME \
        --jobID=$JOBPID  \
        --algo='VQE_fs' \
        --molecule='H2' \
        --nlayer=1 \
        --omega=0 \
        --basis='sto-3g' \
        --bondlen=0 \
        --refstate='HF' \
        --device='CPU' \
        --lb=0.5 \
        --ub=0.7 \
        --step=0.1 \



