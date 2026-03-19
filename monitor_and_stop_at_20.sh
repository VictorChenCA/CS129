#!/bin/bash
OUTPUT_FILE="Inference_PRM/data_experiments/thinkprm_synth_75.jsonl"
TARGET=20

echo "Monitoring until $TARGET examples..."
echo "Press Ctrl+C to stop monitoring (process will continue)"

while true; do
    if [ -f "$OUTPUT_FILE" ]; then
        COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo 0)
        echo -ne "\rCollected: $COUNT/$TARGET examples"
        
        if [ "$COUNT" -ge "$TARGET" ]; then
            echo -e "\n\n✓ Reached $COUNT examples! Stopping..."
            pkill -f "generate_thinkprm_training_data"
            sleep 1
            FINAL=$(wc -l < "$OUTPUT_FILE")
            echo "✓ Final count: $FINAL examples saved"
            break
        fi
    fi
    sleep 3
done
