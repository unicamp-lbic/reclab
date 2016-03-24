# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 18:20:41 2016

@author: thalita

script para gerar tabela com tempos e performance no latex
Para um so model_size. Repete a tabela pra outros.

RS    Tempo Treinam.    Tempo Rec.   P@20    RMSE
nome   xxx $\pm$ zz ...

"""

import expdb
import pandas as pd
import config
import numpy as np

configs = [
config.BMF5fold,
config.BMFLSH5fold,
config.BMFRP5fold,
config.BMFRPLSH5fold,
config.MixedConfig(config.BMFRP5fold.copy(), config.LinReg.copy(),
                   'dim_red', [str(n) for n in np.arange(0.1,0.31,0.02)])]


# for i, _ in enumerate(configs):
#     configs[i].set_par('database', 'TestDB')
edb = expdb.ExperimentDB()

def make_section():
    global configs, fields, fmts, table
    for conf in configs:
        eid = edb.get_id(conf)
        if eid is not None:
            line = ''
            for i_f, f in enumerate(fields):
                if i_f == 0:
                    s = ''
                    if pd.notnull(edb.db['Ens_type'].loc[eid].values[0]):
                        s += edb.db['Ens_type'].loc[eid].values[0] + ' '
                        s = s.replace('Ensemble', ' Ensemble')\
                            .replace('LinReg', 'Lin. Reg.')\
                            .replace('Rating', ' Rating')\
                            .replace('List', ' List')\
                            .replace('WAvg', 'W. Avg.')\

                    s += edb.db[f].loc[eid].values[0].replace('recommender', '')
                    if edb.db['algorithm'].loc[eid].values[0] == 'LSH':
                        s += '+LSH'
                else:
                    v = edb.db[f].loc[eid]
                    s = fmts[i_f]%(v.mean()) + ' $\pm$ ' + fmts[i_f]%(v.std())
                line += s + ' & '
            line = line[:-2]
            line += '\\\\ \n'
            table += line
        else:
            print('Experiment not found', conf.as_dict())
#%%
fields = ['RS_type', 'train_time', 'rec_time', 'P@20_valid', 'RMSE_valid']
fmts = [None, '%d', '%d', '%0.3f', '%0.3f']
table = '\hline \n'
for model_size in ['1','0.5','0.3']:
    table += '\multicolumn{5}{|c|}{$l=' + (1-int(float(model_size)))*model_size \
        + '|U|$} \\\\ \n'
    for i_c, c in enumerate(configs):
        configs[i_c].set_par('model_size', model_size)
    make_section()
    table += '\hline \n'
header = '\hline\nMétodo & Tempo treinamento & Tempo recomendação & P@20 & RMSE\\\\\hline\n'
table = header + table
print(table)


