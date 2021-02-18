#!/bin/bash -e

BASE_DIR="/home/darius/ClinicalUserEmbeddings/"
OUTPUT_DIR="/media/data_1/darius/"
DATA_DIR="$OUTPUT_DIR/data"
SCIBERT_DIR="/media/data_1/darius/models/scibert_scivocab_uncased/"
mkdir -p "$OUTPUT_DIR/models/"

cd "$BASE_DIR/scripts" 

python finetune_on_pregenerated.py \
	--pregenerated_data "$DATA_DIR/pregen_epochs/128/" \
	--output_dir "$OUTPUT_DIR/models/pretrained_baseline_clinical_BERT_1_epoch_128/" \
	--bert_model "$SCIBERT_DIR" \
	--do_lower_case \
	--reduce_memory \
	--epochs 1 \
	--train_batch_size 8 \
	--seed 123

# python finetune_on_pregenerated.py \
# 	--pregenerated_data "$DATA_DIR/pregen_epochs/512/" \
# 	--output_dir "$OUTPUT_DIR/models/pretrained_baseline_clinical_BERT_1_epoch_512/" \
# 	--bert_model "$OUTPUT_DIR/models/pretrained_baseline_clinical_BERT_1_epoch_128/" \
# 	--do_lower_case \
# 	--epochs 1 \
# 	--train_batch_size 16\
# 	--seed 123

# rm -rf "$OUTPUT_DIR/models/pretrained_baseline_clinical_BERT_1_epoch_128/"
