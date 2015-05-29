# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:07:20 2015

@author: thalita
"""

import numpy as np

def _get_zero_mean_matrix(matrix, along='users'):
    rows, cols = matrix.shape
    if along=='users':
        mean_vals = np.zeros(rows)
        for i in range(rows):
            non_zero = [elem for elem in matrix[i,:] if elem > 0]
            if non_zero != []:
                mean_vals[i] = \
                    sum(non_zero)/float(len(non_zero))
            matrix[i,:] -= mean_vals[i]
    else: # Along Items
        mean_vals = np.zeros(rows)
        for i in range(cols):
            non_zero = [elem for elem in matrix[:,i] if elem > 0]
            if non_zero != []:
                mean_vals[i] = \
                    sum(non_zero)/float(len(non_zero))
            matrix[:,i] -= mean_vals[i]

    return matrix, mean_vals