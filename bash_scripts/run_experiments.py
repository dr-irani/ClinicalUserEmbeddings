import os
import sys
import subprocess
import shlex
import numpy as np
from numpy.random import RandomState
from tqdm import trange

cols = ['Acute cerebrovascular disease',
        'Chronic kidney disease',
        'Congestive heart failure; nonhypertensive',
        'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
        'Shock'
        ]

std_models = ['baseline_clinical_BERT_1_epoch_512',
              'adv_clinical_BERT_gender_1_epoch_512']

model = 'baseline_clinical_BERT_1_epoch_512'
dfname = 'phenotype_all'
# t = 'inhosp_mort'

# file name, col names, models
# tasks = [('inhosp_mort', ['inhosp_mort'],  model),
#         ('phenotype_all', cols, model),
#          ('phenotype_first', cols, model) ]
tasks = [('inhosp_mort', ['inhosp_mort'], model)]


num_seeds = 100
randomizer = RandomState(1)

for t in cols:
    for i in trange(num_seeds):
        seed = randomizer.randint(10000)
        print(f'\nFinetuning {dfname} data on target {t} using seed {seed}...')
        subprocess.call(shlex.split(
            'bash finetune_on_target.sh "%s" "%s" "%s" "%d"' % (dfname, model, t, seed)))


# for dfname, targetnames, model in tasks:
#     for t in targetnames:
#         print(f'Finetuning {dfname} data with {t} task...')
#         # for c,m in enumerate(models):
#         subprocess.call(shlex.split(
#             'bash finetune_on_target.sh "%s" "%s" "%s"' % (dfname, model, t)))
