# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:35:58 2015

@author: thalita

plots for experiment 1
"""

from utils import read_results, plot_metric, plot_metrics, pd_select
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

outdir = '/home/thalita/Dropbox/Mestrado/RelatorioParcial/relatorio_tex/fig/'
#%%
result = []
for i in range(1,4+1):
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
#%%
select = {'RStype': 'AvgRatingEnsemble',
          'RPtype': 'sparse',
          #'mincoverage': 1 ,
          'threshold': 3 ,
          #'offlinekNN': 'False',
          #'neighbortype': 'user',
          #'algorithm': 'brute',
          #'metric': 'cosine'
          }
print(pd_select(result, select))



plot_metrics(all_metrics,
             varpar=('nneighbors', [5, 20, 40]),
             across=('nprojections', 'num. de projeções'),
             dataframe=result, select=select,
             labelfmt='%d vizinhos',
             suptitle='Recomendação Ensemble BMF+RP esparsa - BMF 100% e limiar 3')


#%%
select = {'RStype': 'AvgRatingEnsemble',
          'RPtype': 'sparse',
          #'mincoverage': 1 ,
          'threshold': 3 ,
          #'offlinekNN': 'False',
          #'neighbortype': 'user',
          #'algorithm': 'brute',
          #'metric': 'cosine'
          }
select_BMF = dict(select)
del select_BMF['RPtype']
select_BMF['RStype'] = 'BMFrecommender'
select_BMF.update({'mincoverage': 1 ,
                   'offlinekNN': 'False',
                   'neighbortype': 'user',
                   'algorithm': 'brute',
                   'metric': 'cosine'})
data = pd_select(result, select_BMF)
data['RStype'] = 'AvgRatingEnsemble'
data['RPtype'] = 'sparse'
data['nprojections'] = 0
tmp = pd.concat((result, data), ignore_index=True)

plot_metrics(all_metrics,
             varpar=('nprojections', [0, 5, 10, 20, 30]),
             across=('nneighbors', 'num. de vizinhos'),
             dataframe=tmp, select=select,
             labelfmt='%d projeções',
             suptitle='Recomendação Ensemble BMF+RP esparsa - BMF 100% e limiar 3')
plt.savefig(outdir + 'BMFRP_avgrating_sparse.eps')
#%%
select = {'RStype': 'AvgRatingEnsemble',
          'RPtype': 'gaussian',
          #'mincoverage': 1 ,
          'threshold': 3 ,
          #'offlinekNN': 'False',
          #'neighbortype': 'user',
          #'algorithm': 'brute',
          #'metric': 'cosine'
          }
print(pd_select(result, select))
plot_metrics(all_metrics,
             varpar=('nneighbors', [20]),
             across=('nprojections', 'num. de projeções'),
             dataframe=result, select=select,
             labelfmt='%d vizinhos',
             suptitle='Recomendação Ensemble BMF+RP gaussiana - BMF 100% e limiar 3')

plt.savefig(outdir + 'BMFRP_avgrating_gaussian.eps')
#%%
select = {'RStype': 'AvgRatingEnsemble',
          'nneighbors': 20,
          #'mincoverage': 1 ,
          'threshold': 3 ,
          #'offlinekNN': 'False',
          #'neighbortype': 'user',
          #'algorithm': 'brute',
          #'metric': 'cosine'
          }
print(pd_select(result, select))
plot_metrics(all_metrics,
             varpar=('RPtype', ['sparse', 'gaussian']),
             across=('nprojections', 'num. de projeções'),
             dataframe=result, select=select,
             labelfmt='RP %s',
             suptitle='Recomendação Ensemble BMF+RP esparsa vs. RP gaussiana - BMF 100% e limiar 3')