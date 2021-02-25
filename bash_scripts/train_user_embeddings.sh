#!/bin/bash -e

BASE_DIR="/home/darius/ClinicalUserEmbeddings/"
OUTPUT_DIR="/media/data_1/darius/data"
MODEL_DIR="/media/data_1/darius/models/pretrained_baseline_clinical_BERT_1_epoch_512"

cd "$BASE_DIR/scripts/"

python train_user_embeddings.py \
	--train_df "$OUTPUT_DIR/df_grouped.pkl" \
	--col_name "BERT_sents20" \
	--output_dir "$OUTPUT_DIR/pregen_epochs/user/" \
	--bert_model "$MODEL_DIR" \
	--epochs_to_generate 1 \
	--max_seq_len 512