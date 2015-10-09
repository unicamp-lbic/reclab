#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:46 2015

@author: thalita
"""

import argparse
import time
from datetime import datetime
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
    parser.add_argument('-c', '--config',
                        help='Configuration setting for this run \
    (see valid_configs in config.py)')
    parser.add_argument('--id',
                        help='experiment id to erase (user with clear exp)')
    parser.add_argument('-s','--sweep', help='do param sweep')
    parser.add_argument('-v','--values', help='values for param sweep')
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
        conf = config.valid_configs[args.config]
    except KeyError:
        raise KeyError('Invalid configuration setting')

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
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                try:
                    conf.__getattribute__(par)
                    conf.__setattr__(par, value)
                except AttributeError:
                    if par in conf.RS_args:
                        conf.RS_args[par] = value
                    else:
                        raise ValueError('Invalid config param')
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
    will need --config, --sweep, --values, --ensemble_config
    '''
    if args.ensemble is not None:
        try:
            ensemble_conf = config.valid_ensemble_configs[args.ensemble]
        except KeyError:
            raise KeyError('Invalid ensemble configuration setting')

        run_ensemble(args, conf, ensemble_conf, exp_db)
    elif args.sweep is not None:
        '''
        Check for a param sweep
        '''
        run_sweep(args, conf, exp_db)
    else:
        run_exp(args, conf, exp_db)


def run_sweep(args, conf, exp_db):
    values = args.values.split(',')
    try:
        values = [int(x) for x in values]
    except ValueError:
        try:
            values = [float(x) for x in values]
        except ValueError:
            pass
    for v in values:
        if args.sweep in conf.__dict__:
            conf.__setattr__(args.sweep, v)
        elif args.sweep in conf.RS_args:
            conf.RS_args[args.sweep] = v
        else:
            raise ValueError('Parameter not present in specified cofiguration')
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

    if args.ensemble is not None:
        RS = evalu.load_recommendations(FOLD_PATH)
        return RS

    RS = conf.RS_type(**conf.RS_args)
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
        metrics = evalu.Metrics(split, FOLD_PATH)
        metrics.def_test_set(args.set)
        metrics.error_metrics()
        metrics.list_metrics(conf.threshold)
        for arg, val in metrics.metrics.items():
            exp_db.set_fold_arg_val(EXP_ID, fold, arg, val)

    else:
        raise ValueError('Invalid action')


    return RS


def run_ensemble(args, conf, ensemble_conf, exp_db):
    values = args.values.split(',')
    try:
        values = [int(x) for x in values]
    except ValueError:
        try:
            values = [float(x) for x in values]
        except ValueError:
            pass
    RS_list = []
    for v in values:
        if args.sweep in conf.__dict__:
            conf.__setattr__(args.sweep, v)
        elif args.sweep in conf.RS_args:
            conf.RS_args[args.sweep] = v
        else:
            raise ValueError('Parameter not present in specified cofiguration')
        RS_folds = run_exp(args, conf, exp_db)
        RS_list.append(RS_folds)

    '''
    Create expID for ensemble exp
    '''
    params = conf.asdict()
    params.update(ensemble_conf.as_dict())
    params[args.sweep] = values
    EXP_ID = exp_db.get_id_dict(params)
    if EXP_ID is None:
        EXP_ID = time.strftime('%Y%m%d%H%M%S')
        exp_db.add_experiment(EXP_ID, params)

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
        ens = ensemble_conf.RS_type(**ensemble_conf.RS_args)
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
            metrics = evalu.Metrics(split, FOLD_PATH)
            metrics.def_test_set(args.set)
            metrics.error_metrics()
            metrics.list_metrics(conf.threshold)
            for arg, val in metrics.metrics.items():
                exp_db.set_fold_arg_val(EXP_ID, fold, arg, val)

def get_timestamp():
    dt = datetime.now()
    ms = int(int(dt.microsecond)/1e4)
    return dt.strftime('%Y%m%d%H%M%S') + str(ms)

if __name__=='__main__':
    main()
