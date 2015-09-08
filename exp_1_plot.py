# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:35:58 2015

@author: thalita

plots for experiments 1 and 2
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
"""
Plots of metrics by number of neighbors
"""

select = {'RStype': 'BMFrecommender',
          'threshold': 3,
          'offlinekNN': 'False',
          'neighbortype': 'user',
          'algorithm': 'brute',
          'metric': 'cosine'}

plot_metrics(all_metrics,
             varpar=('mincoverage',[0.6, 0.8, 1]),
             across=('nneighbors', 'num. de vizinhos'),
             dataframe=result, select=select,
             labelfmt='BMF %d%%', labelmul=100,
             suptitle='Recomendação BMF baseada em usuário - limiar=3')
plt.savefig(outdir + 'BMF_coverage_nneighbors.eps')
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