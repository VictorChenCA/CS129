#!/bin/bash
# Monitor script for single task training

while true; do
    clear
    echo "=== Single Task Training Progress ==="
    echo ""
    tmux capture-pane -t singletask_training -p | grep -E "Epoch|Step|loss=|train_loss|eval_loss" | tail -5
    echo ""
    echo "=== GPU Memory ==="
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{used=$1; free=$2; util=$3; printf "Used: %d MiB, Free: %d MiB, GPU Util: %s%%\n", used, free, util}'
    echo ""
    echo "=== Recent Log Output ==="
    tail -3 /home/maxrod/CS224n-1/singletask_training.log 2>/dev/null || echo "Log file not found yet"
    echo ""
    echo "=== Checking for errors ==="
    tail -10 /home/maxrod/CS224n-1/singletask_training.log 2>/dev/null | grep -i "error\|exception\|traceback" | tail -3 || echo "No errors found"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 30
done
