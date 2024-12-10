#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: source run_pipeline.bash --csv_file_name <file.csv>"
    echo "  --csv_file_name   Specify the path to the CSV file"
    return 1
}


# Function to get value of a parameter from config file
get_param() {
    grep "^$1:" heartwise.config | cut -d':' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# Read parameters from config file
model_device=$(get_param "model_device")
data_path=$(get_param "data_path")
config_path=$(get_param "config_path")
batch_size=$(get_param "batch_size")
output_folder=$(get_param "output_folder")
hugging_face_api_key_path=$(get_param "hugging_face_api_key_path")
use_x3d=$(get_param "use_x3d")
use_mvit=$(get_param "use_mvit")
num_workers=$(get_param "num_workers")
preprocessing_workers=$(get_param "preprocessing_workers")
video_path=$(get_param "video_path")
eval_granularity=$(get_param "eval_granularity")

# Overwrite mode by command line argument
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --csv_file_name)
            if [[ -n $2 && ! $2 =~ ^-- ]]; then
                csv_file_name="$2"
                shift 2
            else
                echo "Error: --csv_file requires a non-empty argument."
                usage
                return 1
            fi
            ;;
        --help|-h)
            usage
            return 0
            ;;
        *)
            echo "Error: Unknown parameter passed: $1"
            usage
            return 1
            ;;
    esac
done

# Function to run the pipeline with given mode
run_pipeline() {
    python main.py \
        --model_device $model_device \
        --data_path $data_path"/"$csv_file_name \
        --config_path $config_path \
        --batch_size $batch_size \
        --output_folder $output_folder \
        --hugging_face_api_key_path $hugging_face_api_key_path \
        --use_x3d $use_x3d \
        --use_mvit $use_mvit \
        --video_path $video_path \
        --num_workers $num_workers \
        --preprocessing_workers $preprocessing_workers \
        --eval_granularity $eval_granularity
}

# Main execution based on mode
run_pipeline