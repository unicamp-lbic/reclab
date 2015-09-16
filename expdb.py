# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:57:59 2015

@author: thalita
"""
import os
import numpy as np
import pandas as pd
from utils import pd_select

DBFILE = 'experiments.db'
def load_experiments_db(dbfile=DBFILE):
    exp_db = ExperimentDB()
    if not os.path.isfile(dbfile):
        exp_db.new_db()
        exp_db.save_db(dbfile)
    else:
        exp_db.load_db(dbfile)
    return exp_db

def save_experiments_db(exp_db, dbfile=DBFILE):
    exp_db.save_db(dbfile)

class ExperimentDB(object):
    def __init__(self):
        self.db = None

    def load_db(self, dbfile):
        self.db = pd.read_pickle(dbfile)

    def new_db(self):
        self.db = pd.DataFrame()

    def save_db(self, dbfile):
        self.db.to_pickle(dbfile)

    def _get_entries(self, conf):
        df = pd_select(self.db, conf.as_dict)
        if  df.shape[0] == 0:
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

    def get_arg_val(self, exp_id, arg_name):
        return self.get_fold_arg_val(exp_id, 0, arg_name)

    def get_fold_arg_val(self, exp_id, fold, arg_name):
        try:
            val = self.db.get_value((exp_id,fold), arg_name)
            if np.isnan(val):
                return None
            else:
                return val
        except KeyError:
            return None

    def set_arg_val(self, exp_id, arg_name, arg_val):
        self.db.set_value(exp_id, arg_name, arg_val)

    def set_fold_arg_val(self, exp_id, fold, arg_name, arg_val):
        self.db.set_value((exp_id, fold), arg_name, arg_val)

    def add_experiment(self, exp_id, conf):
        data = [conf.as_dict()]
        index = [(exp_id, 0)]
        index = pd.MultiIndex.from_tuples(index, names=['exp_id', 'fold'])
        df = pd.DataFrame(data, index=index)
        self.db.append(df)
