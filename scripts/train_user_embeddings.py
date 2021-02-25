import json
import pickle
import logging
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm, trange
from pathlib import Path
from random import random, randrange, randint, shuffle, choice
from transformers import BertTokenizer
from pregenerate_training_data import DocumentDatabase, truncate_seq_pair, create_masked_lm_predictions, getGroups
import Constants


log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    subject_id = doc_database.subject_ids[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    count = 0
    while i < len(document):
        segment = document[i]['tokens']
        current_chunk.append(segment)
        current_length += len(segment)
        groups_a = document[i]['groups']
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                random_document = None
                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(
                        current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        doc_b_last = random_document[j]
                        tokens_b.extend(doc_b_last['tokens'])
                        if len(tokens_b) >= target_b_length:
                            break
                    groups_b = doc_b_last['groups']
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    groups_b = groups_a
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                try:
                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1
                except AssertionError:
                    count += 1
                    i += 1
                    continue
                    # TODO: verify that instances aren't lost

                tokens = ["[CLS]"] + [f"[usr_{subject_id}]"] + tokens_a + \
                    ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(
                    len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                    'groups_a': groups_a,
                    'groups_b': groups_b}
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    print(f'lost sentences: {count}')
    return instances


def prepare_docs(args, tokenizer, data):
    if len(args.categories) > 0:
        for i in args.categories:
            # make sure each category is present
            assert((df['category'] == i).sum() > 0)
        df = df[df['category'].isin(args.categories)]
        if df.shape[0] == 0:
            raise Exception('dataframe is empty after subsetting!')

    if len(args.drop_group) > 0:
        print('Records before dropping: %s' % len(df))
        for i in Constants.drop_groups[args.drop_group]:
            df = df[df[args.drop_group] != i]
        print('Records after dropping: %s' % len(df))

    docs = DocumentDatabase()
    for _, row in tqdm(data.iterrows()):
        doc = []
        groups = getGroups(row)
        for _, line in enumerate(row[args.col_name]):
            sample = {
                'tokens': tokenizer.tokenize(line),
                'groups': groups
            }
            doc.append(sample)
        docs.add_document(row.subject_id, doc)
    return docs


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--train_df', type=str, required=True)
    parser.add_argument('--col_name', type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True)
    parser.add_argument("--do_whole_word_mask", action="store_true", default=True,
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument('--categories', type=str, nargs='+',
                        dest='categories', default=[])
    parser.add_argument('--drop_group', type=str, default='',
                        help='name of adversarial protected group to drop classes for')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True)
    vocab_list = list(tokenizer.vocab.keys())

    prepare_docs_path = Path(
        '/media/data_1/darius/data/prepared_user_docs.pkl')

    if not prepare_docs_path.is_file():
        logging.info('Loading grouped dataframe...')
        df = pd.read_pickle(args.train_df)

        logging.info('Dataframe loaded. Preparing docs...')
        docs = prepare_docs(args, tokenizer, df)
        del df

        with prepare_docs_path.open('wb') as f:
            pickle.dump(docs, f)
        logging.info('Documents prepared!')
    else:
        with prepare_docs_path.open('rb') as f:
            logging.info(f'Loading documents from {prepare_docs_path}...')
            docs = pickle.load(f)
            logging.info('Documents loaded!')

    args.output_dir.mkdir(exist_ok=True, parents=True)
    subject_ids = [f'[usr_{subj_id}]' for subj_id in docs.subject_ids]
    tokenizer.add_tokens(subject_ids)
    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        epoch_filename = args.output_dir / f"epoch_{epoch}.json"
        num_instances = 0
        with epoch_filename.open('w') as epoch_file:
            for doc_idx in trange(len(docs), desc="Document"):
                doc_instances = create_instances_from_document(
                    docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                    masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                    whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
                doc_instances = [json.dumps(instance)
                                 for instance in doc_instances]
                for instance in doc_instances:
                    epoch_file.write(instance + '\n')
                    num_instances += 1
        metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
        with metrics_file.open('w') as metrics_file:
            metrics = {
                "num_training_examples": num_instances,
                "max_seq_len": args.max_seq_len
            }
            metrics_file.write(json.dumps(metrics))

    tokenizer.save_vocabulary(args.bert_model)


if __name__ == '__main__':
    main()
