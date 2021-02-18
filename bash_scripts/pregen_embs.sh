#!/bin/sh -e

BASE_DIR="/home/darius/ClinicalUserEmbeddings"
OUTPUT_DIR="/media/data_1/darius"
DATA_DIR="$OUTPUT_DIR/data"
cd "$BASE_DIR/scripts"
mkdir -p "$DATA_DIR/pregen_embs/"
emb_method='cat4'
model="baseline_clinical_BERT_1_epoch_512"
target="inhosp_mort_100"
experiment="gender"
echo $model

# for target in inhosp_mort phenotype_first phenotype_all; do
echo $target
# for model in baseline_clinical_BERT_1_epoch_512 adv_clinical_BERT_1_epoch_512; do
    # echo $model
python pregen_embeddings.py \
    --df_path "$DATA_DIR/finetuning/$target"\
    --model_path "$OUTPUT_DIR/models/$model" \
    --output_path "${DATA_DIR}/pregen_embs/pregen_${model}_${emb_method}_${target}_${experiment}" \
    --emb_method $emb_method
    # done
# done

