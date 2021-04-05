import pandas as pd
import os
import re
import logging
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import re
from collections import defaultdict
pd.set_option('display.max_columns', None)

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

folder = '/media/data_1/darius/models/finetuned'

numFolds = 1000

sheets = ['overall', 'gender', 'language', 'insurance', 'ethnicity']

allowed_models = {'overall': ['baseline'],
                  'gender': ['baseline'],
                  'language': ['baseline'],
                  'insurance': ['baseline'],
                  'ethnicity': ['baseline']}
model = 'baseline'


def gap_significant(lower, upper):
    return (((lower < 0) & (upper < 0)) | ((lower > 0) & (upper > 0)))


def add_to_dict(gap_dict, model, sheet, name, num_sig, num_favor):
    if model not in gap_dict:
        gap_dict[model] = {}
    if sheet not in gap_dict[model]:
        gap_dict[model][sheet] = {}
    gap_dict[model][sheet][name] = [num_sig, num_favor]


def add_gap(model, gap, sheet, res, gap_infos_naive):
    res[model+'_' + gap+'_sig'] = multipletests(
        res[model+'_'+gap + '_p'], alpha=0.05, method="fdr_bh")[0]
    add_to_dict(gap_infos_naive, model, sheet, gap, res[model+'_' + gap + 'naive_sig'].astype(
        int).sum(), (res.loc[res[model + '_' + gap + 'naive_sig'], model + '_' + gap] > 0).astype(int).sum())


def get_seeds(finetuned_dir):
    p = Path(folder)
    seeds = [f.name.split('_seed')[1] for f in p.glob('*_seed[0-9]*')]

    return seeds


def get_target_name(mname):
    target = None
    if 'inhosp_mort' in mname:
        target = 'inhosp_mort'
    elif 'phenotype' in mname:
        mname = mname.split('seed')[0]
        name = re.findall(r'.*512_(?:lambda1_)*(.*)', mname)[0]
        if name.endswith('_gs'):
            name = name[:-3]
        name = name.replace('_', ' ')
        if 'phenotype_all' in mname:
            target = 'phenotype_all_%s' % name
        else:
            target = 'phenotype_first_%s' % name

    assert(target)
    return target


def populate_df(*, df, res, idx, model, columns, multi=False):
    for i in columns:
        col = model + '_' + columns[i]
        res.loc[idx, col] = df.loc[i, 'avg']
        res.loc[idx, col + '_p'] = df.loc[i, 'p']
        res.loc[idx, col + '_favor'] = df.loc[i, 'favor']

        res.loc[idx, col + 'lowerCI'] = df.loc[i, '2.5%']
        res.loc[idx, col + 'upperCI'] = df.loc[i, '97.5%']
        if multi and ('dgap_' in col or 'egap_' in col):
            res.loc[idx, col + 'naive_sig'] = gap_significant(
                df.loc[i, '2.5%'], df.loc[i, '97.5%'])
        else:
            res.loc[idx, col + 'naive_sig'] = gap_significant(
                df.loc[i, '2.5%'], df.loc[i, '97.5%'])


def process_files():
    seeds = get_seeds(folder)
    dfs_list = []
    gap_infos_naive_list = []
    for seed in tqdm(seeds):
        dfs = {}
        gap_infos_naive = {}
        for sheet in sheets:
            res = pd.DataFrame()
            for root, dirs, files in os.walk(folder):
                for d in dirs:
                    mname = d
                    if f'_seed{seed}' not in mname:
                        continue
                    file = Path(root) / d / 'results.xlsx'
                    if not file.is_file():
                        logging.error(
                            f'Cannot find {file.parents[0]} results file. Skipping...')
                        continue

                    target = get_target_name(mname)
                    logging.info(f'Target {target} with seed {seed}')

                    if sheet == 'overall':
                        df = pd.read_excel(os.path.join(
                            root, file), index_col=0, sheet_name='all')
                        columns = ['all_auroc', 'all_auprc',
                                   'all_recall', 'all_class_true_count']
                        for i in columns:
                            res.loc[f'{target}-{seed}', model +
                                    '_' + i] = df.loc[i, 'avg']
                            res.loc[f'{target}-{seed}', model +
                                    '_' + i+'lowerCI'] = df.loc[i, '2.5%']
                            res.loc[f'{target}-{seed}', model + '_' +
                                    i+'upperCI'] = df.loc[i, '97.5%']

                    elif sheet == 'gender':
                        df = pd.read_excel(os.path.join(
                            root, file), index_col=0, sheet_name='gender')
                        columns = {
                            'gender=="M"_dgap_max': 'Parity Gap (M-F)',
                            'gender=="M"_egap_positive_max': 'Recall Gap',
                            'gender=="M"_egap_negative_max': 'Specificity Gap'
                        }

                        populate_df(df=df, res=res, idx=target,
                                    model=model, columns=columns)

                    elif sheet == 'language':
                        df = pd.read_excel(os.path.join(
                            root, file), index_col=0, sheet_name='language_to_use')
                        columns = {'language_to_use=="English"_dgap_max': 'Parity Gap (E-O)',
                                   'language_to_use=="English"_egap_positive_max': 'Recall Gap',
                                   'language_to_use=="English"_egap_negative_max': 'Specificity Gap'}

                        populate_df(df=df, res=res, idx=target,
                                    model=model, columns=columns)

                    elif sheet == 'insurance':
                        df = pd.read_excel(os.path.join(
                            root, file), index_col=0, sheet_name='insurance')
                        columns = []
                        for i in ['Medicare', 'Private', 'Medicaid']:
                            for j in ['dgap_max', 'egap_positive_max', 'egap_negative_max']:
                                columns.append(
                                    'insurance=="%s"_%s' % (i, j)
                                )

                        columns = {k: k.replace('insurance==', '')
                                   for k in columns}
                        populate_df(df=df, res=res, idx=target,
                                    model=model, columns=columns, multi=True)

                    elif sheet == 'ethnicity':
                        df = pd.read_excel(os.path.join(
                            root, file), index_col=0, sheet_name='ethnicity_to_use')
                        columns = []
                        for i in ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC/LATINO', 'OTHER']:
                            for j in ['dgap_max', 'egap_positive_max', 'egap_negative_max']:
                                columns.append(
                                    'ethnicity_to_use=="%s"_%s' % (i, j)
                                )
                        columns = {k: k.replace('ethnicity_to_use==', '')
                                   for k in columns}
                        populate_df(df=df, res=res, idx=target,
                                    model=model, columns=columns, multi=True)

            if sheet == 'gender':
                for m in allowed_models[sheet]:
                    for i in ('Parity Gap (M-F)', 'Recall Gap', 'Specificity Gap'):
                        add_gap(m, i, sheet, res, gap_infos_naive)

            if sheet == 'language':
                for m in allowed_models[sheet]:
                    for i in ('Parity Gap (E-O)', 'Recall Gap', 'Specificity Gap'):
                        add_gap(m, i, sheet, res, gap_infos_naive)

            if sheet == 'insurance':
                for m in allowed_models[sheet]:
                    for g in ['Medicare', 'Private', 'Medicaid']:
                        for i in ('"%s"_' % g + t for t in ['dgap_max', 'egap_positive_max', 'egap_negative_max']):
                            add_gap(m, i, sheet, res, gap_infos_naive)

            if sheet == 'ethnicity':
                for m in allowed_models[sheet]:
                    for g in ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC/LATINO', 'OTHER']:
                        for i in ('"%s"_' % g + t for t in ['dgap_max', 'egap_positive_max', 'egap_negative_max']):
                            add_gap(m, i, sheet, res, gap_infos_naive)
            res = res.reset_index()
            dfs[sheet] = res.sort_index()
        dfs_list.append(dfs)
        gap_infos_naive_list.append(gap_infos_naive)

        return dfs_list, gap_infos_naive_list


def display_tables(df):
    for i in ['gender', 'language', 'ethnicity', 'insurance']:
        temp = df.T.xs(i, level=1).dropna(axis=1)
        temp = temp.apply(lambda x: x.apply(lambda y: str(
            y[0]) + ' (' + "{:.0%}".format(y[1]/y[0]) + ')'), axis=0)
        if i in ['ethnicity', 'insurance']:
            temp = temp.T
            temp['Gap'] = list(
                map(lambda x: list(reversed(re.split(r'"_', x)))[0][:-4], temp.index))
            temp['Group'] = list(
                map(lambda x: list(reversed(re.split(r'"_', x)))[1][1:].lower(), temp.index))
            temp = temp.set_index(['Gap', 'Group']).sort_index()
        elif i == 'gender':
            columns = ['Recall Gap', 'Parity Gap (M-F)', 'Specificity Gap']
            temp = temp[columns]
        elif i == 'language':
            columns = ['Recall Gap', 'Parity Gap (E-O)', 'Specificity Gap']
            temp = temp[columns]
        display(temp)
        if i in ['ethnicity', 'insurance']:
            temp = temp[['baseline']].reset_index()
            temp = temp.pivot_table(
                values='baseline', index='Group', columns='Gap', aggfunc=lambda x: x)
            temp = temp[['egap_positive', 'dgap', 'egap_negative']]
            if i == 'ethnicity':
                temp = temp.loc[['white', 'black',
                                 'hispanic/latino', 'asian', 'other']]
            elif i == 'insurance':
                temp = temp.loc[['medicare', 'private', 'medicaid']]
            display(temp)


def main():
    dfs_list, gap_infos_naive_list = process_files()

    for gap_infos_naive in gap_infos_naive_list:
        dict_of_df = {k: pd.DataFrame(v) for k, v in gap_infos_naive.items()}
        naive_df = pd.concat(dict_of_df, axis=1, sort=False)
        display_tables(naive_df)


if __name__ == '__main__':
    main()
