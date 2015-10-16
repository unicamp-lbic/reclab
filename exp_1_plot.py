# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:35:58 2015

@author: thalita

plots for experiments 1 and 2
"""

from utils import pd_select
from plot import read_results, plot_metric, plot_metrics,
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import expdb
from itertools import chain
import config

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

outdir = '/tmp/'
#%%
exp_db = expdb.ExperimentDB()
result = exp_db.db
colors = 'bgrcmyk'
all_metrics = ['P','R','F1','RMSE','MAE']
ir_metrics = ['P','R','F1']
trad_metrics = ['RMSE','MAE']
#%%
"""
Plots of metrics by number of neighbors
"""
select = config.BMF5fold.as_dict()
del select ['n_neighbors']
select['database']='TestDB'
print(select)

#%%
plot_metrics(all_metrics,
             suptitle='Recomendação BMF baseada em usuário - limiar=3',
             varpar=('min_coverage',[1]),
             across=('n_neighbors', 'num. de vizinhos'),
             dataframe=result, select=select,
             split='valid', atN=20,
             labelfmt='BMF %d%%', labelmul=100)
#plt.savefig(outdir + 'BMF_coverage_nneighbors.eps')
#%%
select = {'RStype': 'BMFrecommender',
          'threshold': 3,
          'offlinekNN': 'False',
          'neighbortype': 'item',
          'algorithm': 'brute',
          'metric': 'cosine'}

plot_metrics(all_metrics,
             varpar=('mincoverage',[0.6, 0.8, 1]),
             across=('nneighbors', 'num. de vizinhos'),
             dataframe=result, select=select,
             labelfmt='BMF %d%%', labelmul=100,
             suptitle='Recomendação BMF baseada em item - limiar=3')
#plt.savefig(outdir + 'BMF_coverage_nneighbors_item.eps')
#%%
for cov in [0.6, 0.8, 1]:
    select = {'RStype': 'BMFrecommender',
              'mincoverage': cov ,
              'offlinekNN': 'False',
              'neighbortype': 'user',
              'algorithm': 'brute',
              'metric': 'cosine'}

    plot_metrics(all_metrics,
                 varpar=('threshold',[x for x in range(0,5)]),
                 across=('nneighbors', 'num. de vizinhos'),
                 dataframe=result, select=select,
                 labelfmt='BMF t=%d',
                 suptitle='Recomendação BMF baseada em usuário - %d%% cobertura'%(100*cov))

    plt.savefig(outdir + 'BMF_threshold_nneighbors_mincoverage_'+str(cov)+'.eps')
#%%
for cov in [0.6, 0.8, 1]:
    select = {'RStype': 'BMFrecommender',
              'mincoverage': cov ,
              'offlinekNN': 'False',
              'neighbortype': 'item',
              'algorithm': 'brute',
              'metric': 'cosine'}
    plot_metrics(all_metrics,
                 varpar=('threshold',[x for x in range(0,5)]),
                 across=('nneighbors', 'num. de vizinhos'),
                 dataframe=result, select=select,
                 labelfmt='BMF t=%d',
                 suptitle='Recomendação BMF baseada em item - %d%% cobertura'%(100*cov))
#%%
select = {'RStype': 'BMFrecommender',
          'mincoverage': 1 ,
          'threshold': 3,
          'neighbortype': 'user',
          'metric': 'cosine'}
data = result
for key, value in select.items():
    data = data[data[key] == value]
plot_metrics(all_metrics,
             varpar=('algorithm', ['brute', 'LSH']),
             across=('nneighbors', 'num. de vizinhos'),
             dataframe=result, select=select,
             labelfmt='kNN %s',
             suptitle='Recomendaçao BMF+LSH Forest - BMF 100% e limiar 3')
plt.savefig(outdir + 'BMFLSH_nneighbors.eps')