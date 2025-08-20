#!/bin/bash

# Weight ablation batch runner
# Usage: nohup ./run_weight_ablation_batch.sh config1.json config2.json config3.json > batch.log 2>&1 &

if [ $# -eq 0 ]; then
    echo "Usage: nohup $0 <config1.json> [config2.json] [config3.json] ... > batch.log 2>&1 &"
    exit 1
fi

for config in "$@"; do
    if [ ! -f "$config" ]; then
        echo "Config file not found: $config"
        continue
    fi
    
    config_name=$(basename "$config" .json)
    log_file="weight_ablation_${config_name}.log"
    
    echo "Starting ablation for $config_name..."
    python ablation/run_weight_ablation.py \
        --data data/dataset_arcagi.json \
        --weight-range 0,1 \
        --step-size 0.01 \
        --quiet \
        --config "$config" 2>&1 | tee "$log_file"
    
    echo "Completed $config_name"
done

echo "All jobs started"