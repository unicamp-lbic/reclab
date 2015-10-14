# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:28 2015

@author: thalita

Config file
"""
import abc
from collections import defaultdict
import recommender as rec
import ensemble as ens
import pandas as pd
import data


class BaseConfig(object):
    __metaclass__ = abc.ABCMeta
    def as_dict(self):
        d = self.__dict__.copy()
        keys = list(d.keys())
        for key in keys:
            if d[key] is None:
                del d[key]


        return d

class Config(BaseConfig):
    def __init__(self, database, RS_type, RS_args,
                 nfolds, pct_hidden, threshold,
                 per_user=True, is_MF=False):
        self.database = data.base.STD_DB_NAMES[database]
        self.RS_type = RS_type
        self.RS_args = RS_args
        self.nfolds = nfolds
        self.per_user = per_user
        self.pct_hidden = pct_hidden
        self.threshold = threshold
        self.is_MF = is_MF
        if is_MF:
            self.MF_type = RS_type.__MF_type__
            self.MF_args = RS_type.__MF_args__(RS_args)

    def as_dict(self):
        d = BaseConfig.as_dict(self)
        del d['RS_args']
        d['RS_type'] = d['RS_type'].__name__
        d.update(self.RS_args)
        return d

class EnsembleConfig(BaseConfig):
    def __init__(self, Ens_type, Ens_args):
        self.Ens_type = Ens_type
        self.Ens_args = Ens_args

    def as_dict(self):
        d = BaseConfig.as_dict(self)
        del d['Ens_args']
        d['Ens_type'] = d['Ens_type'].__name__
        d.update(self.Ens_args)
        return d

dummy5fold = Config(
    database='ml100k',
    RS_type=rec.DummyRecommender,
    RS_args={},
    is_MF=False,
    nfolds=5,
    threshold=3,
    pct_hidden=0.2
)

IB5fold = Config(
    database='ml100k',
    RS_type=rec.ItemBased,
    RS_args={'n_neighbors': 20,
             'model_size': 2,
             'algorithm': 'brute',
             'metric': 'cosine',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=5,
    is_MF=False,
    threshold=3,
    pct_hidden=0.2
)

UB5fold = Config(
    database='ml100k',
    RS_type=rec.UserBased,
    RS_args={'n_neighbors': 20,
             'algorithm': 'brute',
             'metric': 'correlation',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=5,
    is_MF=False,
    threshold=3,
    pct_hidden=0.2
)

BMF5fold = Config(
    database='ml100k',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 20,
             'model_size': 2,
             'neighbor_type': 'user',
             'algorithm': 'brute',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 3,
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=5,
    is_MF=True,
    threshold=3,
    pct_hidden=0.2
)

BMFLSH5fold = Config(
    database='ml100k',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 20,
             'model_size': 2,
             'neighbor_type': 'user',
             'offline_kNN': True,
             'algorithm': 'LSH',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 3,
             'weighting': 'none',
             'n_estimators': 10,
             'n_candidates': 2},
    nfolds=5,
    is_MF=True,
    threshold=3,
    pct_hidden=0.2
)

BMFRP5fold = Config(
    database='ml100k',
    RS_type=rec.BMFRPrecommender,
    RS_args={'RP_type': 'sparse',
             'dim_red': 0.8,
             'n_neighbors': 20,
             'model_size': 2,
             'neighbor_type': 'user',
             'algorithm': 'brute',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 3,
             'offline_kNN': True,
             'weighting': 'tf-idf'},
    nfolds=5,
    is_MF=True,
    threshold=3,
    pct_hidden=0.2
)

WAvg = EnsembleConfig(
    Ens_type=ens.WAvgRatingEnsemble,
    Ens_args={}
)

LinReg = EnsembleConfig(
    Ens_type=ens.LinRegRatingEnsemble,
    Ens_args={'regularization': 1.0,
              'l1_ratio': 0.5  }
)

'''
Dictionary of valid configuration settings
'''
valid_configs = {
    'BMF5fold': BMF5fold,
    'BMFLSH5fold': BMFLSH5fold,
    'BMFRP5fold': BMFRP5fold,
    'IB5fold': IB5fold,
    'dummy5fold': dummy5fold,
    'UB5fold': UB5fold
}

valid_ensemble_configs = {
    'WAvg': WAvg,
    'LinReg': LinReg
}