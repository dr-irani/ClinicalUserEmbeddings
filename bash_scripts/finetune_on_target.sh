#!/bin/bash
#SBATCH --partition t4 
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --output=finetune_%A.out
#SBATCH --mem 60gb

# $1 - target type {inhosp_mort, phenotype_first, phenotype_all}
# $2 - BERT model name {baseline_clinical_BERT_1_epoch_512, adv_clinical_BERT_1_epoch_512}
# $3 - target column name within the dataframe, ex: "Shock", "any_acute"
# $4 - seed
# $5 - user embedding experiment name

# set -e 
# source activate hurtfulwords

BASE_DIR="/home/darius/HurtfulWords"
OUTPUT_DIR="/media/data_1/darius"
DATA_DIR="${OUTPUT_DIR}/data"

cd "$BASE_DIR/scripts"

python finetune_on_target.py \
	--df_path "${DATA_DIR}/finetuning/$1" \
	--model_path "${OUTPUT_DIR}/models/$2" \
	--fold_id 9 10\
	--target_col_name "$3" \
	--output_dir "${OUTPUT_DIR}/models/finetuned/${1}_${2}_${3}_seed${4}/" \
	--freeze_bert \
	--train_batch_size 32 \
	--pregen_emb_path "${DATA_DIR}/pregen_embs/pregen_${2}_cat4_${1}" \
	--task_type binary \
	--other_fields age sofa sapsii sapsii_prob oasis oasis_prob \
        --gridsearch_classifier \
        --gridsearch_c \
        --emb_method cat4 \
	--overwrite \
	--seed $4
