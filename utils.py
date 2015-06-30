# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:07:20 2015

@author: thalita
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def oneD(array):
    return np.array(np.array(array).squeeze(), ndmin=1)

def _get_zero_mean_matrix(matrix, along='users'):
    rows, cols = matrix.shape
    if along == 'users':
        mean_vals = np.zeros(rows)
        for i in range(rows):
            non_zero = [elem for elem in matrix[i, :] if elem > 0]
            if non_zero != []:
                mean_vals[i] = \
                    sum(non_zero)/float(len(non_zero))
            matrix[i, :] -= mean_vals[i]
    else:  # Along Items
        mean_vals = np.zeros(rows)
        for i in range(cols):
            non_zero = [elem for elem in matrix[:, i] if elem > 0]
            if non_zero != []:
                mean_vals[i] = \
                    sum(non_zero)/float(len(non_zero))
            matrix[:, i] -= mean_vals[i]

    return matrix, mean_vals


def read_result(fname, path='', meanstd=True):
    splitted = fname[:fname.find('pct')-4].split('_')
    params = {'RStype': splitted[0]}
    for i in range(1, len(splitted)-1, 2):
        value = splitted[i+1]
        try:
            value = float(value)
        except ValueError:
            pass
        params[splitted[i]] = value

    with open(path + fname, 'r') as f:
        header = f.readline()
    header = header.replace('#', '').replace('"', '').replace(' ', '')\
        .replace('\n', '').split(',')
    header = header + [h+'(std)' for h in header]
    result = np.loadtxt(path + fname, delimiter=',', ndmin=2)
    if meanstd:
        result = np.hstack((result.mean(axis=1), result.std(axis=1)))
    result = dict(zip(header, result))
    result.update(params)
    return result


def read_results(path='', meanstd=True):
    fnames = [f for f in os.listdir(path) if f.find('test.txt') > -1]
    result = []
    for fname in fnames:
        result.append(read_result(fname, path, meanstd=meanstd))
    return result


def plot_metric(metric, varpar, across, dataframe, select,
                labelfmt='%s', labelmul=1):
    data = dataframe
    for key, value in select.items():
        data = data[data[key] == value]

    varpar_name, varpar_values = varpar
    across, across_label = across
    for value in varpar_values:
        this_data = data[data[varpar_name] == value]
        this_data.sort(across, inplace=True)
        x = this_data[across].values
        y = this_data[this_data[varpar_name] == value][metric].values
        plt.plot(x, y, marker='+', label=labelfmt % (value*labelmul))
    plt.legend(loc='best', fontsize='small', framealpha=0.5)
    plt.title(metric)
    plt.xlabel(across_label)


def plot_metrics(metrics, varpar, across, dataframe, select, labelfmt='%s',
                 labelmul=1, suptitle=None):

    width = int(np.ceil(np.sqrt(len(metrics))))
    height = int(np.ceil(len(metrics)/width))
    plt.figure(figsize=(4*width,3*height))
    for i, metric in enumerate(metrics):
        plt.subplot(height, width, i+1)
        plot_metric(metric, varpar, across, dataframe, select,
                    labelfmt, labelmul)
    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle)

