#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:46 2015

@author: thalita
"""

import argparse
import time
from datetime import datetime
import numpy as np
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
    parser.add_argument('-c', '--config', action='append',
                        help='Configuration setting for this run \
    (see valid_configs in config.py)')
    parser.add_argument('--id',
                        help='experiment id to erase (user with clear exp)')
    parser.add_argument('-s','--sweep', action='append',
                        help='--sweep param=val1,val2,val3 do param sweep')

    parser.add_argument('--folds',
                        help='specific folds to perform action on, comma-separated')
    parser.add_argument('--setpar', action='append', help='--setpar parname=value')
    parser.add_argument('--ensemble', help='--ensemble ENSEMBLE_CONFIG. \
    Do ensemble. Use with --sweep --config --values')
    parser.add_argument('--set', help='--set test|valid, to use with metrics action')
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
    exp_db = expdb.ExperimentDB()

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
        conf = config.valid_configs[args.config[0]]
    except KeyError:
        raise KeyError('Invalid configuration setting')

    '''
    Try to load ensemble config if applicable
    '''
    if args.ensemble is not None:
        try:
            ensemble_conf = config.valid_ensemble_configs[args.ensemble]
        except KeyError:
            raise KeyError('Invalid ensemble configuration setting')

    '''
    Check for --setpar
    '''
    if args.setpar is not None:
        for par_val in args.setpar:
            if par_val.find('=') < 0:
                raise ValueError('Must use --setpar parname=value')
            par, value = tuple(par_val.split('='))
            if value is not None:
                try:
                    conf.set_par(par, value)
                except AttributeError:
                    if args.ensemble is not None:
                        ensemble_conf.set_par(par, value)
            else:
                raise ValueError('Must use --setpar parname=value')
    '''
    process clear_conf command
    '''
    if args.action == 'clear_conf':
        exp_db.clear_conf(conf)
        exit()

    '''
    Check for ensemble action
    will need --config, --sweep par_name=par_values, --ensemble
    '''
    if args.ensemble is not None:
        run_ensemble(args, conf, ensemble_conf, exp_db)
    elif args.sweep is not None:
        '''
        Check for a param sweep
        '''
        run_sweep(args, conf, exp_db)
    else:
        run_exp(args, conf, exp_db)


def run_sweep(args, conf, exp_db):
    for arg in args.sweep:
        sweep = arg.split('=')[0]
        values = arg.split('=')[1].split(',')
        for v in values:
            conf.set_par(sweep, v)
            run_exp(args, conf, exp_db)


def run_exp(args, conf, exp_db):
    '''
    Create exp ID if necessary and corresponding result folder
    Add entry to experiments database
    '''
    EXP_ID = exp_db.get_id(conf)
    if EXP_ID is None:
        EXP_ID = get_timestamp()
        exp_db.add_experiment(EXP_ID, conf)

    RESULT_FOLDER = './results/' + EXP_ID + '/'
    if not os.path.isdir(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER, mode=0o775)

    '''
    Do database split if not done
    Try to get split_fname_prefix from exp_db.
    If not found, do split.
    Anyway, assign split_fname_prefix found/created to this experiment's entry
    '''
    split_fname_prefix = exp_db.get_arg_val(EXP_ID, 'split_fname_prefix', conf)
    if split_fname_prefix is None:
        database = databases.MatrixDatabase(data.dbread(conf.database))
        splitter = ds.CVTestRatingSplitter(nfolds=conf.nfolds,
                                           pct_hidden=conf.pct_hidden,
                                           per_user=conf.per_user,
                                           threshold=conf.threshold)
        split_folder = data.get_db_path(conf.database) \
            + get_timestamp() + '/'
        if not os.path.isdir(split_folder):
            os.makedirs(split_folder, mode=0o775, exist_ok=True)
        split_fname_prefix = splitter.split_save(database, split_folder)
    # Save split fname prefix on this experiment's entry
    for fold in range(conf.nfolds):
        exp_db.set_fold_arg_val(EXP_ID, fold,
                                'split_fname_prefix', split_fname_prefix)

    if args.action == 'split':
        exit()

    '''
    Run experiment
    '''
    RS_list = []
    if args.folds is not None:
        folds = [int(x) for x in args.folds.split(',') if int(x) < conf.nfolds]
    else:
        folds = [x for x in range(conf.nfolds)]
    for fold in folds:
        RS_list.append(run_fold(args, fold, conf, EXP_ID, RESULT_FOLDER,
                                exp_db, split_fname_prefix))

    return RS_list

def run_fold(args, fold, conf, EXP_ID, RESULT_FOLDER, exp_db, split_fname_prefix):
    FOLD_PREFIX =  'fold_%d' % fold
    FOLD_PATH = RESULT_FOLDER + FOLD_PREFIX
    if conf.nfolds == 1:
        split = evalu.load_split(split_fname_prefix)
    else:
        split = evalu.load_split(split_fname_prefix, fold)

    RS = conf.RS_type(**conf.RS_args)
    if args.ensemble is not None:
        if args.action == 'train':
            evalu.load_model(RS, FOLD_PATH, split)
        elif args.action == 'test' or args.action == 'metrics':
            RS = evalu.load_recommendations(FOLD_PATH)
        return RS

    if args.action == 'train':
        # Gen/Load MF if applicable
        if conf.is_MF:
            MF_file_prefix = exp_db.get_fold_arg_val(EXP_ID, fold, 'MF_file_prefix', conf)
            mf_dt = exp_db.get_fold_arg_val(EXP_ID, fold, 'MF_time', conf)
            if MF_file_prefix is None:
                MF_file_prefix = FOLD_PATH
                t0 = time.time()
                evalu.gen_mf(split, MF_file_prefix, RS)
                mf_dt = time.time() - t0
            exp_db.set_fold_arg_val(EXP_ID, fold, 'MF_time', mf_dt)
            exp_db.set_fold_arg_val(EXP_ID, fold, 'MF_file_prefix', MF_file_prefix)
            RS = evalu.load_mf(MF_file_prefix, RS)
        # train and save
        t0 = time.time()
        evalu.train_save(RS, split, FOLD_PATH)
        tr_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'train_file_prefix', FOLD_PATH)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'train_time', tr_dt)

    elif args.action == 'test':
        t0 = time.time()
        evalu.test_save(RS, FOLD_PATH, split)
        tst_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'test_file_prefix', FOLD_PATH)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'test_time', tst_dt)

    elif args.action == 'metrics':
        metrics = evalu.Metrics(split, filepath=FOLD_PATH)
        metrics.def_test_set(args.set)
        metrics.error_metrics()
        metrics.list_metrics(conf.threshold)
        for arg, val in metrics.metrics.items():
            exp_db.set_fold_arg_val(EXP_ID, fold, arg, val)

    else:
        raise ValueError('Invalid action')


    return RS


def run_ensemble(args, conf, ensemble_conf, exp_db):
    # ensemble will only use one sweep for now
    sweep = args.sweep[0].split('=')[0]
    values = args.sweep[0].split('=')[1].split(',')
    for v in values:
        conf.set_par(sweep, v)
        RS_folds = run_exp(args, conf, exp_db)
        RS_list.append(RS_folds)

    '''
    Create expID for ensemble exp
    '''
    params = conf.as_dict()
    params.update(ensemble_conf.as_dict())
    params[args.sweep] = 'varpar'
    params['varpar'] = args.sweep
    params['varpar_values'] = values
    EXP_ID = exp_db.get_id_dict(params)
    if EXP_ID is None:
        EXP_ID = get_timestamp()
        exp_db.add_experiment_dict(EXP_ID, params)

    RESULT_FOLDER = './results/' + EXP_ID + '/'
    if not os.path.isdir(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    '''
    Get split_fname_prefix from exp_db.
    '''
    single_exp_id = exp_db.get_id(conf)
    split_fname_prefix = \
        exp_db.get_arg_val(single_exp_id, 'split_fname_prefix', conf)

    for fold in range(conf.nfolds):
        ens = ensemble_conf.Ens_type(**ensemble_conf.Ens_args)
        exp_db.set_arg_val(EXP_ID, 'split_fname_prefix', split_fname_prefix)
        FOLD_PREFIX =  'fold_%d' % fold
        FOLD_PATH = RESULT_FOLDER + FOLD_PREFIX
        if conf.nfolds == 1:
            split = evalu.load_split(split_fname_prefix)
        else:
            split = evalu.load_split(split_fname_prefix, fold)

        for i, v in enumerate(values):
            ens.RS_list.append(RS_list[i][fold])

        if args.action == 'train':
            t0 = time.time()
            evalu.ensemble_train_save(ens, FOLD_PATH, split)
            tr_dt = time.time() - t0
            exp_db.set_fold_arg_val(EXP_ID, fold, 'train_file_prefix', FOLD_PATH)
            exp_db.set_fold_arg_val(EXP_ID, fold, 'train_time', tr_dt)

        elif args.action == 'test':
            t0 = time.time()
            evalu.ensemble_test_save(ens, FOLD_PATH, split)
            tst_dt = time.time() - t0
            exp_db.set_fold_arg_val(EXP_ID, fold, 'test_file_prefix', FOLD_PATH)
            exp_db.set_fold_arg_val(EXP_ID, fold, 'test_time', tst_dt)

        elif args.action == 'metrics':
            metrics = evalu.Metrics(split, filepath=FOLD_PATH)
            metrics.def_test_set(args.set)
            metrics.error_metrics()
            metrics.list_metrics(conf.threshold)
            for arg, val in metrics.metrics.items():
                exp_db.set_fold_arg_val(EXP_ID, fold, arg, val)
        else:
            raise ValueError('Invalid action')


def get_timestamp():
    dt = datetime.now()
    ms = int(int(dt.microsecond)/1e4)
    return dt.strftime('%Y%m%d%H%M%S') + str(ms)

def run_plot(args, exp_db):
    '''
    plot will accept multiple configs
    will need a --type arg
    for each config it may have one sweep
    a metric plot will need --xaxis param_name
    a PR plot only uses configs and sweeps
    '''
    for conf_arg, sweep_args in zip(args.config, args.sweep):
        '''
        Try to load configuration settings
        '''
        try:
            conf = config.valid_configs[conf_arg]
        except KeyError:
            raise KeyError('Invalid configuration setting')

        '''
        parse sweep params
        '''
        sweep = sweep_args.split('=')[0]
        values = sweep_args.split('=')[1].split(',')
        for v in values:
            conf.set_par(sweep, v)
            plt.figure()
            if args.type == 'metrics':
                metric_names = evalu.Metrics.ir_metric_names(args.set, args.atN) + \
                    evalu.Metrics.error_metric_names(args.set)
                select = conf.as_dict()
                del select[args.xaxis]
                ids = exp_db.get_ids_dict(select)
                plot_single_metric(df, x_axis, metric, **plotargs)

            elif args.type == 'PR':
                id_ = exp_db.get_id(conf)

    '''
    Try to load ensemble config if applicable
    '''
    if args.ensemble is not None:
        try:
            ensemble_conf = config.valid_ensemble_configs[args.ensemble]
        except KeyError:
            raise KeyError('Invalid ensemble configuration setting')


if __name__=='__main__':
    main()
