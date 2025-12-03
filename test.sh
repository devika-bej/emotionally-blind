#!/bin/bash

input_dir="./ib_dataset"
output_dir="./ib_dataset_output"

mkdir -p "$output_dir"

for video_file in "$input_dir"/*; do
    filename=$(basename "$video_file")
    python3 ib_simulation.py "$video_file" "$output_dir/$filename"
done