# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:57:59 2015

@author: thalita
"""
import os
import pandas as pd
from utils import pd_select
from subprocess import call


DBFILE = 'experiments.db'
ARGS = {'MF':'MF_file_prefix',
        'split': 'split_fname_prefix',
        'train':  'train_file_prefix',
        'test': 'test_file_prefix',
        'metrics': 'metrics_file_prefix'}

def load_experiments_db(dbfile=DBFILE):
    exp_db = ExperimentDB()
    if not os.path.isfile(dbfile):
        exp_db.new_db(dbfile)
        exp_db.save_db()
    else:
        exp_db.load_db(dbfile)
    return exp_db

def save_experiments_db(exp_db):
    exp_db.save_db()

class ExperimentDB(object):
    def __init__(self):
        self.db = None
        self.dbfile = None

    def load_db(self, dbfile):
        self.db = pd.read_pickle(dbfile)
        self.dbfile = dbfile

    def new_db(self, dbfile):
        self.db = pd.DataFrame()
        self.dbfile = dbfile
        self.save_db()

    def save_db(self):
        self.db.to_pickle(self.dbfile)
        self.db.to_csv(self.dbfile+'.csv', na_rep=' ')

    def _get_entries(self, conf):
        df = pd_select(self.db, conf.as_dict())
        if df is None:
            return None
        elif df.shape[0] == 0:
            return None
        else:
            return df

    def get_id(self, conf):
        df = self._get_entries(conf)
        if df is not None:
            '''
            df.index returns a MultiIndex object
            df.index.get_level_values('exp_id')[0] returns the exp_id
            '''
            return df.index.get_level_values('exp_id')[0]

    def get_arg_val(self, exp_id, arg_name, conf=None):
        return self.get_fold_arg_val(exp_id, 0, arg_name, conf)

    def get_fold_arg_val(self, exp_id, fold, arg_name, conf=None):
        try:
            val = self.db.get_value((exp_id, fold), arg_name)
            if pd.isnull(val) and conf is not None:
                '''
                did not find value for this experiment
                try to locate compatible experiment for specific args
                '''
                if arg_name == ARGS['split'] or arg_name == ARGS['MF']:
                    if conf.nfolds == 1:
                        select = {'database': conf.database,
                                  'nfolds': conf.nfolds,
                                  'per_user': conf.per_user,
                                  'pct_hidden': conf.pct_hidden}
                    else:
                        select = {'database': conf.database,
                                  'nfolds': conf.nfolds,
                                  'per_user': conf.per_user}
                    if arg_name == ARGS['MF']:
                        select.update({'MF_type': conf.MF_type})
                        select.update(conf.MF_args)
                    df = pd_select(self.db, select)
                    if df is None:
                        val = None
                    elif df.shape[0] == 0:  # did not find compatible experiment
                        val = None
                    else:
                        exp_id = df.index.get_level_values('exp_id')[0]
                        val = self.db.get_value((exp_id, fold), arg_name)
                        if pd.isnull(val):
                            val = None
                else:
                    val = None
            # check if path exists
            # if it does not, refered experiment was deleted
            # clear entry
            if val is not None and not pd.isnull(val):
                if not os.path.exists(os.path.split(val)[0]):
                    val = None
                    self.db.set_value((exp_id, fold), arg_name, '')
            return val
        except KeyError:
            return None

    def set_arg_val(self, exp_id, arg_name, arg_val):
        self.db.set_value(exp_id, arg_name, arg_val)
        self.save_db()

    def set_fold_arg_val(self, exp_id, fold, arg_name, arg_val):
        self.db.set_value((exp_id, fold), arg_name, arg_val)
        self.save_db()

    def add_experiment(self, exp_id, conf):
        data = [conf.as_dict()]*conf.nfolds
        index = [(exp_id, i) for i in range(conf.nfolds)]
        index = pd.MultiIndex.from_tuples(index, names=['exp_id', 'fold'])
        df = pd.DataFrame(data, index=index)
        self.db = self.db.append(df)
        self.save_db()

    def clear_experiment(self, exp_id):
        call(["trash", './results/'+ exp_id + '/'])
        self.db.drop(exp_id, inplace=True)
        self.save_db()

    def clear_conf(self, conf):
        self.clear_experiment(self.get_id(conf))

    def __str__(self):
        return self.db.__str__()

    def print(self):
        with pd.option_context('display.max_rows', self.db.shape[0],
                               'display.max_columns', self.db.shape[1]):
            print(self.db)


