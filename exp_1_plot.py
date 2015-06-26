# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:35:58 2015

@author: thalita

plots for experiment 1
"""

from utils import read_results
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'results/exp_1_results/'
result = read_results(path, meanstd=False)

colors = 'bgrcmyk'
result = pd.DataFrame.from_dict(result)

#%%
"""
Plots of metrics by number of neighbors
"""

for n, nntype in enumerate(['user']):
    plt.figure(n, figsize=(12,6))
    for i, metric in enumerate(['P','R','F1','RMSE','MAE']):
        for j, min_coverage in enumerate([0.6, 0.8, 1]):
            plt.subplot(2, 3, i+1)
            data = result[result.mincoverage == min_coverage]\
                [result.threshold==3]\
                [result.neighbortype==nntype]
            data.sort('nneighbors', inplace=True)
            x = data['nneighbors'].values
            y = data[metric].values
            plt.plot(x,y, marker='+', color=colors[j],
                     label='BMF %d%%'%(100*min_coverage))
            plt.legend(loc='best', fontsize='small', framealpha=0.5)
            plt.title(metric)
            plt.xlabel('nbr of neighbors')
    plt.tight_layout()
    plt.suptitle('BMF %s-based'%nntype)
    plt.show()
#%%
