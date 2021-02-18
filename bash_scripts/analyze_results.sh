#!/bin/sh
#SBATCH --partition cpu 
#SBATCH -c 2
#SBATCH --output bootstrap%A.log
#SBATCH --mem 50gb

# set -e
# source activate hurtfulwords

BASE_DIR="/home/darius/HurtfulWords"
OUTPUT_DIR="/media/data_1/darius/"
cd "$BASE_DIR/scripts"

python analyze_results.py \
	--models_path "${OUTPUT_DIR}/models/finetuned/" \
	--set_to_use "test" \
	--bootstrap \
