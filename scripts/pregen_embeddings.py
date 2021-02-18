import Constants
from run_classifier_dataset_utils import InputExample, convert_examples_to_features
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
import pickle
import numpy as np
import pandas as pd
from utils import MIMICDataset, extract_embeddings, get_emb_size
from torch.utils import data
import torch
import argparse
import sys
import os
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser('''Given a BERT model and a dataset with a 'seqs' column, outputs a pickled dictionary
                                 mapping note_id to 2D numpy array, where each array is num_seq x emb_dim''')
parser.add_argument(
    '--df_path', help='must have the following columns: seqs, num_seqs, and note_id either as a column or index')
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--emb_method', default='last', const='last', nargs='?',
                    choices=['last', 'sum4', 'cat4'], help='how to extract embeddings from BERT output')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.model_path)
config = BertConfig.from_pretrained(args.model_path, output_hidden_states=True)
model = BertModel.from_pretrained(args.model_path, config=config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f'Using {device} with {n_gpu} GPUs')

# if n_gpu > 1:
#     model = torch.nn.DataParallel(model)

print('Reading dataframe...')
df = pd.read_pickle(args.df_path)
if 'note_id' in df.columns:
    df = df.set_index('note_id')


def convert_input_example(note_id, text, seqIdx, subj_id, gender):
    return InputExample(guid='%s-%s' % (note_id, seqIdx), subject_id=subj_id,
                        gender=gender, text_a=text, text_b=None, label=0, group=0, other_fields=[])


print('Converting input examples...')
examples = [convert_input_example(idx, i, c, row.subject_id, row.gender)
            for idx, row in tqdm(df.iterrows()) for (c, i) in enumerate(row.seqs)]
print('Featurizing...')
features = convert_examples_to_features(
    examples, Constants.MAX_SEQ_LEN, tokenizer, output_mode='classification')

generator = data.DataLoader(MIMICDataset(
    features, 'train', 'classification'),  shuffle=True,  batch_size=32)

print('Generate embeddings...')
EMB_SIZE = get_emb_size(args.emb_method)


def get_embs(generator):
    model.eval()
    embs = {str(idx): np.zeros(shape=(
        row['num_seqs'], EMB_SIZE), dtype=np.float32) for idx, row in df.iterrows()}
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, _, _, guid, _ in tqdm(generator):
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            outputs = model(input_ids, token_type_ids=segment_ids,
                            attention_mask=input_mask)
            bert_out = extract_embeddings(outputs[2], args.emb_method)

            for c, i in enumerate(guid):
                note_id, seq_id = i.split('-')
                emb = bert_out[c, :].detach().cpu().numpy()
                embs[note_id][int(seq_id), :] = emb
    return embs


model.to(device)
model_name = os.path.basename(os.path.normpath(args.model_path))
pickle.dump(get_embs(generator), open(args.output_path, 'wb'))
