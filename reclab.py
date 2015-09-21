#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:46 2015

@author: thalita
"""

import argparse
import time
import config
import os
import evaluation as evalu
import data
import datasplit as ds
import expdb
import databases
from subprocess import call


def main():
    '''
    Parse command line params
    '''
    parser = argparse.ArgumentParser(description='Run recommender training/evaluation')
    parser.add_argument('action', help='train, test, metrics, clear_db, \
    clear_exp --id EXP_ID, clear_conf -c CONFIG, show_db')
    parser.add_argument('-c', '--config', help='Configuration setting for this run \
    (see valid_configs in config.py)')
    parser.add_argument('--id', help='experiment id to erase (user with clear exp)')
    parser.add_argument('-s','--sweep', help='do param sweep')
    parser.add_argument('-V','--values', help='values for param sweep')
    args = parser.parse_args()

    '''
    Process clear_db command
    '''
    if args.action == 'clear_db':
        call(["trash", expdb.DBFILE])
        exit()

    '''
    Load experiments DB
    '''
    exp_db = expdb.load_experiments_db()

    '''
    process show_db
    '''
    if args.action == 'show_db':
        exp_db.print()
        exit()
    '''
    Process clear exp command
    '''
    if args.action == 'clear_exp':
        exp_id = args.id
        exp_db.clear_experiment(exp_id)
        exit()

    '''
    Try to load configuration settings
    '''
    try:
        conf = config.valid_configs[args.config]
    except KeyError:
        raise KeyError('Invalid configuration setting')

    '''
    process clear_conf command
    '''
    if args.action == 'clear_conf':
        exp_db.clear_conf(conf)
        exit()

    '''
    Check for a param sweep
    '''
    if args.sweep is not None:
        run_sweep(args, conf, exp_db)
    else:
        run_exp(args, conf, exp_db)


def run_sweep(args, conf, exp_db):
    for v in args.values:
        new_conf = conf.copy()
        new_conf.__setattr__(args.sweep, v)
        run_exp(args, new_conf, exp_db)


def run_exp(args, conf, exp_db):
    '''
    Create exp ID if necessary and corresponding result folder
    Add entry to experiments database
    '''
    EXP_ID = exp_db.get_id(conf)
    if EXP_ID is None:
        EXP_ID = time.strftime('%Y%m%d%H%M%S')
        exp_db.add_experiment(EXP_ID, conf)

    RESULT_FOLDER = './results/' + EXP_ID + '/'
    if not os.path.isdir(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    '''
    Do database split if not done
    Try to get split_fname_prefix from exp_db.
    If not found, do split.
    Anyway, assign split_fname_prefix found/created to this experiment's entry
    '''
    split_fname_prefix = exp_db.get_arg_val(EXP_ID, 'split_fname_prefix', conf)
    if split_fname_prefix is None:
        database = databases.MatrixDatabase(data.dbread(conf.database))
        if conf.nfolds == 1:
            splitter = ds.HoldoutRatingSplitter(conf.pct_hidden,
                                                conf.per_user,
                                                conf.threshold)
        else:
            splitter = ds.kFoldRatingSplitter(conf.nfolds, conf.per_user)

        splitter.split(database)
        split_fname_prefix = splitter.save(data.get_db_path(conf.database))

    '''
    Run experiment
    '''
    for fold in range(conf.nfolds):
        run_fold(args, fold, conf, EXP_ID, RESULT_FOLDER, exp_db, split_fname_prefix)


def run_fold(args, fold, conf, EXP_ID, RESULT_FOLDER, exp_db, split_fname_prefix):
    # Save split fname prefix on this experiment's entry
    exp_db.set_arg_val(EXP_ID, 'split_fname_prefix', split_fname_prefix)
    FOLD_PREFIX =  'fold_%d' % fold
    FOLD_PATH = RESULT_FOLDER + FOLD_PREFIX
    if conf.nfolds == 1:
        split = evalu.load_split(split_fname_prefix)
    else:
        split = evalu.load_split(split_fname_prefix, fold)

    RS = conf.RS_type(**conf.RS_args)
    if args.action == 'train':
        # Gen/Load MF if applicable
        if conf.is_MF:
            MF_file_prefix = exp_db.get_fold_arg_val(EXP_ID, fold, 'MF_file_prefix', conf)
            if MF_file_prefix is None:
                MF_file_prefix = FOLD_PATH
                t0 = time.time()
                evalu.gen_mf(split, MF_file_prefix, conf.MF_type, conf.MF_args)
                mf_dt = time.time() - t0
            exp_db.set_fold_arg_val(EXP_ID, fold, 'MF_file_prefix', MF_file_prefix)
            exp_db.set_fold_arg_val(EXP_ID, fold, 'MF_time', mf_dt)
            RS = evalu.load_mf(MF_file_prefix, RS)
        # train and save
        t0 = time.time()
        evalu.train_save(RS, split, FOLD_PATH)
        tr_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'train_file_prefix', FOLD_PATH)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'train_time', tr_dt)

    elif args.action == 'test':
        t0 = time.time()
        evalu.test_save(RS, FOLD_PATH)
        tst_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'test_file_prefix', FOLD_PATH)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'test_time', tst_dt)

    elif args.action == 'metrics':
        metrics = evalu.Metrics(split, FOLD_PATH)
        metrics.error_metrics()
        metrics.list_metrics(conf.threshold)
        for arg, val in metrics.metrics.items():
            exp_db.set_fold_arg_val(EXP_ID, fold, arg, val)

    else:
        raise ValueError('Invalid action')

    '''
    Save modified expdb
    '''
    expdb.save_experiments_db(exp_db)

if __name__=='__main__':
    main()