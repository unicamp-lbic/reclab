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
from copy import deepcopy

class BaseConfig(object):
    __metaclass__ = abc.ABCMeta
    def as_dict(self):
        d = self.__dict__.copy()
        keys = list(d.keys())
        for key in keys:
            if d[key] is None:
                del d[key]
        return d

    @abc.abstractmethod
    def _set_internal_args(self, par, value):
        pass

    def copy(self):
        return deepcopy(self)

    def set_par(self, par, value):
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        try:
            self.__getattribute__(par)
            self.__setattr__(par, value)
        except AttributeError:
            self._set_internal_args(par, value)

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
            self.MF_type = RS_type.__MF_type__.__name__
            self.MF_args = RS_type.__MF_args__(RS_args)

    def get_name(self):
        return self.RS_type.__name__.replace('recommender','')

    def as_dict(self):
        d = BaseConfig.as_dict(self)
        del d['RS_args']
        d['RS_type'] = d['RS_type'].__name__
        d.update(self.RS_args)
        return d

    def _set_internal_args(self, par, value):
        if par in self.RS_args:
            self.RS_args[par] = value
            if self.is_MF:
                self.MF_type = self.RS_type.__MF_type__.__name__
                self.MF_args = self.RS_type.__MF_args__(self.RS_args)
        else:
            raise AttributeError('Invalid config param')


class EnsembleConfig(BaseConfig):
    def __init__(self, Ens_type, Ens_args):
        self.Ens_type = Ens_type
        self.Ens_args = Ens_args

    def get_name(self):
        return self.Ens_type.__name__.replace('Ensemble','Ens')\
            .replace('Rating','R').replace('List','L')

    def as_dict(self):
        d = BaseConfig.as_dict(self)
        del d['Ens_args']
        d['Ens_type'] = d['Ens_type'].__name__
        d.update(self.Ens_args)
        return d

    def _set_internal_args(self, par, value):
        if par in self.Ens_args:
            self.Ens_args[par] = value
        else:
            raise AttributeError('Invalid config param')


class MixedConfig(Config, EnsembleConfig):
    def __init__(self, conf, ens_conf, varpar, varpar_values):
        conf_dict = conf.__dict__
        if conf.is_MF:
            del conf_dict['MF_args']
            del conf_dict['MF_type']
        del conf_dict['RS_args'][varpar]
        Config.__init__(self, **conf_dict)
        EnsembleConfig.__init__(self, **ens_conf.__dict__)
        self.varpar = varpar
        self.varpar_values = str(varpar_values)

    def get_name(self):
        return EnsembleConfig.get_name(self) + '(%s)' % Config.get_name(self)

    def _set_internal_args(self, par, value):
        if par in self.RS_args:
            self.RS_args[par] = value
            if self.is_MF:
                self.MF_type = self.RS_type.__MF_type__
                self.MF_args = self.RS_type.__MF_args__(self.RS_args)
        elif par in self.Ens_args:
            self.Ens_args[par] = value
        else:
            raise AttributeError('Invalid config param')

    def as_dict(self):
        d = BaseConfig.as_dict(self)
        del d['Ens_args']
        d['Ens_type'] = d['Ens_type'].__name__
        d.update(self.Ens_args)
        del d['RS_args']
        d['RS_type'] = d['RS_type'].__name__
        d.update(self.RS_args)
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

dummydelicious = Config(
    database='delicious',
    RS_type=rec.DummyRecommender,
    RS_args={},
    is_MF=False,
    nfolds=1,
    threshold=0,
    pct_hidden=0.2
)

IB5fold = Config(
    database='ml100k',
    RS_type=rec.ItemBased,
    RS_args={'n_neighbors': 20,
             'model_size': 1.0,
             'algorithm': 'brute',
             'metric': 'cosine',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=5,
    is_MF=False,
    threshold=3,
    pct_hidden=0.2
)

IBdelicious = Config(
    database='delicious',
    RS_type=rec.ItemBased,
    RS_args={'n_neighbors': 20,
             'model_size': 1.0,
             'algorithm': 'brute',
             'metric': 'cosine',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=1,
    is_MF=False,
    threshold=0,
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

UBdelicious = Config(
    database='delicious',
    RS_type=rec.UserBased,
    RS_args={'n_neighbors': 20,
             'algorithm': 'brute',
             'metric': 'correlation',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=1,
    is_MF=False,
    threshold=0,
    pct_hidden=0.2
)

BMF5fold = Config(
    database='ml100k',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 30,
             'model_size': 1.0,
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

BMFdelicious = Config(
    database='delicious',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 20,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'algorithm': 'brute',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 0,
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=1,
    is_MF=True,
    threshold=0,
    pct_hidden=0.2
)

BMFLSH5fold = Config(
    database='ml100k',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 30,
             'model_size': 1.0,
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

BMFLSHdelicious = Config(
    database='delicious',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 20,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'offline_kNN': True,
             'algorithm': 'LSH',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 0,
             'weighting': 'none',
             'n_estimators': 10,
             'n_candidates': 2},
    nfolds=1,
    is_MF=True,
    threshold=0,
    pct_hidden=0.2
)

BMFRP5fold = Config(
    database='ml100k',
    RS_type=rec.BMFRPrecommender,
    RS_args={'RP_type': 'sparse',
             'dim_red': 0.5,
             'n_neighbors': 30,
             'model_size': 1.0,
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

BMFRPdelicious = Config(
    database='delicious',
    RS_type=rec.BMFRPrecommender,
    RS_args={'RP_type': 'sparse',
             'dim_red': 0.5,
             'n_neighbors': 20,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'algorithm': 'brute',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 0,
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=1,
    is_MF=True,
    threshold=0,
    pct_hidden=0.2
)

BMFRPLSH5fold = Config(
    database='ml100k',
    RS_type=rec.BMFRPrecommender,
    RS_args={'RP_type': 'sparse',
             'dim_red': 0.5,
             'n_neighbors': 30,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'algorithm': 'LSH',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 3,
             'offline_kNN': True,
             'weighting': 'none',
             'n_estimators': 10,
             'n_candidates': 2},
    nfolds=5,
    is_MF=True,
    threshold=3,
    pct_hidden=0.2
)

BMFRPLSHdelicious = Config(
    database='delicious',
    RS_type=rec.BMFRPrecommender,
    RS_args={'RP_type': 'sparse',
             'dim_red': 0.5,
             'n_neighbors': 20,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'algorithm': 'LSH',
             'metric': 'cosine',
             'min_coverage': 1.0,
             'bin_threshold': 0,
             'offline_kNN': True,
             'weighting': 'none',
             'n_estimators': 10,
             'n_candidates': 2},
    nfolds=1,
    is_MF=True,
    threshold=0,
    pct_hidden=0.2
)


SVD5fold = Config(
    database='ml100k',
    RS_type=rec.SVDrecommender,
    RS_args={'dim': 10,
             'regularization': 0.01},
    nfolds=5,
    is_MF=True,
    threshold=3,
    pct_hidden=0.2
)

SVDdelicious = Config(
    database='delicious',
    RS_type=rec.SVDrecommender,
    RS_args={'dim': 50,
             'regularization': 0.01},
    nfolds=1,
    is_MF=True,
    threshold=0,
    pct_hidden=0.2
)

SVDNN5fold = Config(
    database='ml100k',
    RS_type=rec.SVDrecommender,
    RS_args={'dim': 10,
             'regularization': 0.01,
             'n_neighbors': 30,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'algorithm': 'brute',
             'metric': 'cosine',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=5,
    is_MF=True,
    threshold=3,
    pct_hidden=0.2
)

SVDNNdelicious = Config(
    database='delicious',
    RS_type=rec.SVDrecommender,
    RS_args={'dim': 50,
             'regularization': 0.01,
             'n_neighbors': 30,
             'model_size': 1.0,
             'neighbor_type': 'user',
             'algorithm': 'brute',
             'metric': 'cosine',
             'offline_kNN': True,
             'weighting': 'none'},
    nfolds=1,
    is_MF=True,
    threshold=0,
    pct_hidden=0.2
)

WAvg = EnsembleConfig(
    Ens_type=ens.WAvgRatingEnsemble,
    Ens_args={'keep': 0.5}
)

LinReg = EnsembleConfig(
    Ens_type=ens.LinRegRatingEnsemble,
    Ens_args={'regularization': 1.0,
              'l1_ratio': 0.1,
              'keep': 0.25}
)

Voting = EnsembleConfig(
    Ens_type=ens.MajorityEnsemble,
    Ens_args={'keep': 0.5})

RankSum = EnsembleConfig(
    Ens_type=ens.RankSumEnsemble,
    Ens_args={'keep': 0.5})

'''
Dictionaries of valid configuration settings
'''
valid_configs = {
    'BMF5fold': BMF5fold,
    'BMFLSH5fold': BMFLSH5fold,
    'BMFRPLSH5fold': BMFRPLSH5fold,
    'BMFRP5fold': BMFRP5fold,
    'IB5fold': IB5fold,
    'dummy5fold': dummy5fold,
    'UB5fold': UB5fold,
    'SVD5fold': SVD5fold,
    'BMFdelicious': BMFdelicious,
    'BMFLSHdelicious': BMFLSHdelicious,
    'BMFRPLSHdelicious': BMFRPLSHdelicious,
    'BMFRPdelicious': BMFRPdelicious,
    'IBdelicious': IBdelicious,
    'dummydelicious': dummydelicious,
    'UBdelicious': UBdelicious,
    'SVDdelicious': SVDdelicious
}

valid_ensemble_configs = {
    'WAvg': WAvg,
    'LinReg': LinReg,
    'Voting': Voting,
    'RankSum': RankSum

}
