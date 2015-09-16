# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:28 2015

@author: thalita

Config file
"""
from collections import defaultdict
import recommender as rec
import pandas as pd


class Config(object):
    def __init__(self, database='ml100k', RS_type=rec.BMFrecommender, RS_args={},
                 nfolds=1, is_MF=True):
        self.database = database
        self.RS_type = RS_type
        self.RS_args = RS_args
        self.nfolds = nfolds
        self.is_MF = is_MF


    def as_dict(self):
        d = self.__dict__.copy()
        del d['RS_args']
        d.update(self.RS_args)
        return d



BMF_basic = Config(
    database='ml100k',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 20,
              'neighborhood_type': 'user',
              'offline_kNN': False,
              'algorithm': 'brute',
              'metric': 'cosine',
              'min_coverage': 1.0,
              'threshold': 3},
    nfolds=5,
    is_MF=True
)



'''
Dictionary of valid configuration settings
'''
valid_configs = {
    'DefaultConfig': DefaultConfig
}