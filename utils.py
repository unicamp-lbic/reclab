# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:07:20 2015

@author: thalita
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time


def oneD(array):
    # TODO problably the ndarray.flatten method can do this
    return np.array(np.array(array).squeeze(), ndmin=1)

class timing(object):
    def __init__(self):
        self.t0 = 0
        self.tic()
    def tic(self):
        self.t0 = time()
    def toc(self, text=''):
        if text != '':
            text == ' '
        dt = time()-self.t0
        print(text, 'Time elapsed:',dt,' s')
        self.tic()
        return dt


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


def pd_select(dataframe, select):
    try:
        data = dataframe
        for key, value in select.items():
            data = data[data[key] == value]
            if len(data) == 0:
                return None
        return data
    except KeyError:
        return None
