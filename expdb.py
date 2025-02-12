# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:57:59 2015

@author: thalita
"""
import os
import pandas as pd
from utils import pd_select
from subprocess import call
from lockfile import locked

RESULTS_FOLDER = 'results/'
DBLOCK = RESULTS_FOLDER + 'expdb.lock'
DBFILE = RESULTS_FOLDER + 'experiments.db'

ARGS = {'MF': 'MF_file_prefix',
        'MF_time': 'MF_time',
        'split': 'split_fname_prefix',
        'train':  'train_file_prefix',
        'rec': 'rec_file_prefix',
        'metrics': 'metrics_file_prefix',
        'final_MF': 'MF_file_prefix',
        'MF_time': 'MF_time',
        'final_MF_time': 'final_MF_time',
        'final_train': 'final_train_file_prefix',
        'final_rec': 'final_rec_file_prefix',}


class ExperimentDB(object):
    @locked(DBLOCK)
    def __init__(self, dbfile=DBFILE):
        self.db = None
        self.dbfile = dbfile
        if not os.path.isfile(dbfile):
            self._new_db()
        else:
            self._load_db()

    def _load_db(self):
        self.db = pd.read_pickle(self.dbfile)

    def _new_db(self):
        self.db = pd.DataFrame()
        self._save_db()

    def _save_db(self):
        self.db.to_pickle(self.dbfile)
        self.db.to_csv(self.dbfile+'.csv', na_rep=' ')

    def _get_entries(self, adict):
        df = pd_select(self.db, adict)
        if df is None:
            return None
        elif df.shape[0] == 0:
            return None
        else:
            return df

    @locked(DBLOCK)
    def get_id(self, conf):
        self._load_db()
        df = self._get_entries(conf.as_dict())
        if df is not None:
            '''
            df.index returns a MultiIndex object
            df.index.get_level_values('exp_id')[0] returns the exp_id
            '''
            return df.index.get_level_values('exp_id')[0]

    @locked(DBLOCK)
    def get_id_dict(self, adict):
        self._load_db()
        df = self._get_entries(adict)
        if df is not None:
            '''
            df.index returns a MultiIndex object
            df.index.get_level_values('exp_id')[0] returns the exp_id
            '''
            return df.index.get_level_values('exp_id')[0]

    @locked(DBLOCK)
    def get_ids_dict(self, adict):
        self._load_db()
        df = self._get_entries(adict)
        if df is not None:
            '''
            df.index returns a MultiIndex object
            df.index.get_level_values('exp_id') returns the exp_ids
            '''
            return df.index.get_level_values('exp_id')

    def get_arg_val(self, exp_id, arg_name, conf=None):
        return self.get_fold_arg_val(exp_id, 0, arg_name, conf)

    @locked(DBLOCK)
    def get_fold_arg_val(self, exp_id, fold, arg_name, conf=None):
        self._load_db()
        try:
            val = self.db.get_value((exp_id, fold), arg_name)
            if pd.isnull(val) and conf is not None:
                '''
                did not find value for this experiment
                try to locate compatible experiment for specific args
                '''
                if arg_name == ARGS['split'] or arg_name == ARGS['MF'] or \
                    arg_name == ARGS['final_MF'] or arg_name == ARGS['MF_time']\
                    or arg_name == ARGS['final_MF_time']:
                    select = {'database': conf.database,
                              'nfolds': conf.nfolds,
                              'per_user': conf.per_user,
                              'pct_hidden': conf.pct_hidden}

                    if arg_name == ARGS['MF'] or arg_name == ARGS['final_MF']\
                        or arg_name == ARGS['MF_time']\
                        or arg_name == ARGS['final_MF_time']:
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
                if arg_name.find('file') > -1:
                    if not os.path.exists(os.path.split(val)[0]):
                        val = None
                        self.db.set_value((exp_id, fold), arg_name, '')
            return val
        except KeyError:
            return None

    @locked(DBLOCK)
    def set_arg_val(self, exp_id, arg_name, arg_val):
        self._load_db()
        self.db.set_value(exp_id, arg_name, arg_val)
        self._save_db()

    @locked(DBLOCK)
    def set_fold_arg_val(self, exp_id, fold, arg_name, arg_val):
        self._load_db()
        self.db.set_value((exp_id, fold), arg_name, arg_val)
        self._save_db()

    @locked(DBLOCK)
    def add_experiment(self, exp_id, conf):
        self._load_db()
        data = [conf.as_dict()]*conf.nfolds
        index = [(exp_id, i) for i in range(conf.nfolds)]
        index = pd.MultiIndex.from_tuples(index, names=['exp_id', 'fold'])
        df = pd.DataFrame(data, index=index)
        if len(self.db) == 0:
            self.db = df
        else:
            self.db = self.db.append(df)
        self._save_db()

    @locked(DBLOCK)
    def add_experiment_dict(self, exp_id, adict):
        self._load_db()
        nfolds = adict['nfolds']
        data = [adict]*nfolds
        index = [(exp_id, i) for i in range(nfolds)]
        index = pd.MultiIndex.from_tuples(index, names=['exp_id', 'fold'])
        df = pd.DataFrame(data, index=index)
        self.db = self.db.append(df)
        self._save_db()

    @locked(DBLOCK)
    def clear_experiment(self, exp_id):
        self._load_db()
        call(["trash", './results/'+ exp_id + '/'])
        self.db.drop(exp_id, inplace=True, level=0)
        print('cleared experiment', exp_id)
        self._save_db()

    def clear_conf(self, conf):
        exp_id = self.get_id(conf)
        if exp_id is not None:
            self.clear_experiment(exp_id)

    def __str__(self):
        return self.db.__str__()

    def print(self):
        with pd.option_context('display.max_rows', self.db.shape[0],
                               'display.max_columns', self.db.shape[1]):
            print(self.db)


