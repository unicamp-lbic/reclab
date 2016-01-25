# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:56:21 2015

@author: marcos, thalita

Adapted from mobers project

cria mfs
cira svdrecommender e carrega as mfs
segue fluxo normal pra cada recommender
dps chama ensemble em cima da config do nise
"""
import numpy as np
import config
import reclab
import evaluation as evalu
import data
import datasplit as ds
import expdb
import databases
import time
import os
import utils
import matplotlib.pyplot as plt


class niseCand():
    def __init__(self, n0, n1,norm):
        self.n0 = n0
        self.n1 = n1
        self.imp = ((n0.objs[0]-n1.objs[0])/norm[0]*(n1.objs[1]-n0.objs[1])/norm[1])

    def calcLambdaa(self):
        self.lambdaa = np.linalg.solve([[self.n0.objs[0],self.n0.objs[1],-1],[self.n1.objs[0],self.n1.objs[1],-1],[1,1,0]],[0,0,1])[1]


def do_single_SVD(dim, regularization, fold_id=None, initial_solution=None):
    global database_name
    args = reclab.arg_parsing('trainrecmetrics -c SVD5fold --set valid'.split(' '))
    exp_db = expdb.ExperimentDB()
    conf = config.SVD5fold.copy()

    conf.set_par('regularization', str(regularization))
    conf.set_par('dim', str(dim))
    conf.set_par('database', database_name)
    '''
    Create exp ID if necessary and corresponding result folder
    Add entry to experiments database
    '''
    EXP_ID = exp_db.get_id(conf)
    if EXP_ID is None:
        EXP_ID = reclab.get_timestamp()
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
            + reclab.get_timestamp() + '/'
        if not os.path.isdir(split_folder):
            os.makedirs(split_folder, mode=0o775, exist_ok=True)
        split_fname_prefix = splitter.split_save(database, split_folder)
    # Save split fname prefix on this experiment's entry

    RS_list = []
    for fold in range(conf.nfolds):
        exp_db.set_fold_arg_val(EXP_ID, fold,
                                'split_fname_prefix', split_fname_prefix)

        FOLD_PREFIX =  'fold_%d' % fold
        FOLD_PATH = RESULT_FOLDER + FOLD_PREFIX
        if conf.nfolds == 1:
            split = evalu.load_split(split_fname_prefix)
        else:
            split = evalu.load_split(split_fname_prefix, fold)

        RS = conf.RS_type(**conf.RS_args)

        # Create MF
        MF_file_prefix = exp_db.get_fold_arg_val(EXP_ID, fold, 'MF_file_prefix', conf)
        mf_dt = exp_db.get_fold_arg_val(EXP_ID, fold, 'MF_time', conf)
        try:
            RS = evalu.load_mf(MF_file_prefix, RS)
        except:
            MF_file_prefix = FOLD_PATH
            t0 = time.time()
            evalu.gen_mf(split, MF_file_prefix, RS, listt=initial_solution)
            mf_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'MF_time', mf_dt)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'MF_file_prefix', MF_file_prefix)
        # train and save
        t0 = time.time()
        evalu.train_save(RS, split, FOLD_PATH)
        tr_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'train_file_prefix', FOLD_PATH)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'train_time', tr_dt)
        RS_list.append(RS)
        #rec
        t0 = time.time()
        evalu.rec_save(RS, FOLD_PATH, split)
        tst_dt = time.time() - t0
        exp_db.set_fold_arg_val(EXP_ID, fold, 'rec_file_prefix', FOLD_PATH)
        exp_db.set_fold_arg_val(EXP_ID, fold, 'rec_time', tst_dt)
        # metrics
        metrics = evalu.Metrics(split, filepath=FOLD_PATH)
        metrics.def_test_set(args.set)
        metrics.error_metrics()
        metrics.list_metrics(conf.threshold)
        metrics.coverage_metrics()
        for arg, val in metrics.metrics.items():
            exp_db.set_fold_arg_val(EXP_ID, fold, arg, val)

    if fold_id is None:
        mf = [rs.model for rs in RS_list]
    else:
        mf = RS_list[fold_id].model
    return mf




def nise(fold_id, nSol=50, hVError=0.01, d=50, tol=1e-2):
    plt.figure()
    plt.title('Fold %d' % fold_id)
    plt.xlabel('MSE (training)')
    plt.ylabel('L2 norm')
    mf1 = do_single_SVD(d, 1, fold_id)
    plt.plot(mf1.objs[0], mf1.objs[1], 'bo', ms=6)
    plt.draw()
    mf0 = do_single_SVD(d, 0, fold_id)
    plt.plot(mf0.objs[0], mf0.objs[1], 'bo', ms=6)
    plt.draw()


    initial_norm = [(mf0.objs[0]-mf1.objs[0]),(mf1.objs[1]-mf0.objs[1])]
    efList = [niseCand(mf0, mf1, initial_norm)]
    out = [mf1, mf0]
    sols = 2
    lambdas = set()
    while efList!=[]:
        actual = efList.pop(0)
        if actual.imp>hVError and abs(actual.n0.getReg()-actual.n1.getReg())>tol and sols<nSol:
            actual.calcLambdaa()
            if actual.lambdaa not in lambdas:
                lambdas.add(actual.lambdaa)
                print('lambda', actual.lambdaa, 'n sols', sols)
                mf = do_single_SVD(d, actual.lambdaa, fold_id, initial_solution=out)
                plt.plot(mf.objs[0], mf.objs[1], 'ro')
                plt.draw()

                next = niseCand(mf,actual.n1,initial_norm)
                efList.append(next)

                next = niseCand(actual.n0,mf,initial_norm)
                efList.append(next)

                out.append(mf)
                sols+=1

    return out


if __name__=='__main__':
    global database_name
    database_name = 'TestDB'
    hvError = 0.1e-2 # x % minimal improvement on hypervolume
    d = 5
    tol = 1e-2
    nSol = 50
    RESULT_NISE = './results/nise_%s_dim_%d_nsol_%d_hverror_%f_tol_%f/' \
        % (database_name, d, nSol, hvError, tol)
    if not os.path.exists(RESULT_NISE):
        os.makedirs(RESULT_NISE, mode=0o775)
    for fold in range(5):
        print('Nise fold %d'% fold)
        mf_list = nise(fold, d=d, nSol=nSol, hVError=hvError, tol=tol)
        plt.savefig(RESULT_NISE+'fold_%d_pareto.png' % fold)
        plt.savefig(RESULT_NISE+'fold_%d_pareto.eps')
        text = '--varpar regularization='+','.join([str(mf.lambdaa) for mf in mf_list])
        text += ' --setpar dim=%d' % d
        with open(RESULT_NISE+'fold_%d_lambdas.txt' % fold, 'w') as f:
            f.write(text)
        objs = [(mf.objs, mf.lambdaa) for mf in mf_list]
        utils.to_gzpickle(objs, (RESULT_NISE+'fold _%d_objs' % fold))
