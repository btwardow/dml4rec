#!/usr/bin/env bash

SCRIPT=$1
SCRIPT_NAME=$(basename $SCRIPT)

DATE=$(date +%Y-%m-%d_%H%M%S)
EXPERIMENT_ID="$$_$DATE"
echo "Experiment ID: $EXPERIMENT_ID"

mkdir -p logs/
mkdir -p results/
LOG_FILE="logs/$EXPERIMENT_ID-${SCRIPT_NAME%.*}"".log"
echo "Running script: $SCRIPT"
echo "Logging all output to file: $LOG_FILE"

shift

PYTHONPATH='.' python $SCRIPT $EXPERIMENT_ID $LOG_FILE $@
