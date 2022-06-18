#!/bin/sh

# variables initialization
input_file_path="../input/configs/alibaba_log_transformation_lr_0.01.json"
output_directory_path="../output"

# define auxiliary variables
base_name=$(basename "$input_file_path")
file_name=${base_name%.*}  # no extension
echo "$file_name"

# create output directory for the specific job
mkdir -p ${output_directory_path}/"$file_name"

# run job
PYTHONUNBUFFERED=false PYTHONPATH=$PYTHONPATH:/.. nohup python3 ./app/train_test_model.py --config ${input_file_path} --output ${output_directory_path}/${file_name} > ${output_directory_path}/${file_name}/nohup_log.log &
