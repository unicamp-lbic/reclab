# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:07:20 2015

@author: thalita
"""
import os
import numpy as np


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


def read_result(fname, path=''):
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
    result = np.loadtxt(path + fname, delimiter=',')
    result = np.hstack((result.mean(axis=1), result.std(axis=1)))
    result = dict(zip(header, result))
    result.update(params)
    return result


def read_results(path=''):
    fnames = os.listdir(path)
    result = []
    for fname in fnames:
        result.append(read_result(fname, path))
    return result
