# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:35:07 2015

@author: thalita

plots for exp 3

"""


from utils import read_results, plot_metric, plot_metrics, pd_select
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from itertools import chain

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

outdir = '/tmp/'
#%%
result = []
for i in range(1,3+1):
    path = 'results/exp_%d_results/' % i
    result += read_results(path, meanstd=False)
colors = 'bgrcmyk'
result = pd.DataFrame.from_dict(result)
all_metrics = ['P','R','F1','RMSE','MAE']
ir_metrics = ['P','R','F1']
trad_metrics = ['RMSE','MAE']

#%%
select = {'RStype': 'BMFRPrecommender',
          'RPtype': 'sparse',
          'mincoverage': 1 ,
          'threshold': 3 ,
          'offlinekNN': 'False',
          'neighbortype': 'user',
          'algorithm': 'brute',
          'metric': 'cosine'}

select_BMF = dict(select)
del select_BMF['RPtype']
select_BMF['RStype'] = 'BMFrecommender'
data = pd_select(result, select_BMF)
data['RStype'] = 'BMFRPrecommender'
data['dimred'] = 1
data['RPtype'] = 'sparse'
tmp = pd.concat((result, data), ignore_index=True)

plot_metrics(all_metrics,
             varpar=('dimred', [1, 0.9, 0.8, 0.75, 0.5, 0.25]),
             across=('nneighbors', 'num. de vizinhos'),
             dataframe=tmp, select=select,
             labelfmt='%d%%|D|', labelmul=100,
             suptitle='Recomendação BMF+RP esparsa - BMF 100% e limiar 3')
plt.savefig(outdir + 'BMFRP_sparse_dimred_nneighbors.eps')
#%%
select = {'RStype': 'BMFRPrecommender',
          'RPtype': 'gaussian',
          'mincoverage': 1 ,
          'threshold': 3 ,
          'offlinekNN': 'False',
          'neighbortype': 'user',
          'algorithm': 'brute',
          'metric': 'cosine'}

select_BMF = dict(select)
del select_BMF['RPtype']
select_BMF['RStype'] = 'BMFrecommender'
data = pd_select(result, select_BMF)
data['RStype'] = 'BMFRPrecommender'
data['dimred'] = 1
data['RPtype'] = 'gaussian'
tmp = pd.concat((result, data), ignore_index=True)

plot_metrics(all_metrics,
             varpar=('dimred', [1, 0.9, 0.8, 0.75, 0.5, 0.25]),
             across=('nneighbors', 'num. de vizinhos'),
             dataframe=tmp, select=select,
             labelfmt='%d%%|D|', labelmul=100,
             suptitle='Recomendação BMF+RP gaussiana - BMF 100% e limiar 3')
plt.savefig(outdir + 'BMFRP_gaussian_dimred_nneighbors.eps')

