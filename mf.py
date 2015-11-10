# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:55:09 2015

@author: marcos, thalita

Adapted from mobers project
"""

import numpy as np
from numpy.random import shuffle
import copy
from time import time
import scipy.optimize as opt


class SVD():
    '''
    Generates a MF by minimizing L2-regularized squared error measured on the
    training ratings.
    It is a regularized SVD which works only on available data.
    (regular SVD does not handle missing data)

    Parameters:
    rTrain: list of tuples (rating, user_id, item_id)
    nUsers, nItems: number of users and items in the database
    dim: dimensionality of the latent space
    regularization: regularization strenght
    listt: list of trained models used as initialization (used with nise)
    '''
    def __init__(self, rTrain, nUsers, nItems, dim=50, regularization=0.1,
                 listt=None):
        self.d = dim
        self.lambdaa = regularization

        self.ratings = rTrain
        self.rTrain = rTrain

        self.nU = nUsers
        self.nI = nItems
        self.nR = len(rTrain)

        self.training_time = 0
        self.theta = np.ones(self.nU*self.d + self.nI*self.d)/np.sqrt(self.d)

        self.objs = [self.errorObj(self.theta), self.regObj(self.theta)]
        self.obj = (1-self.lambdaa)*self.objs[0] + (self.lambdaa)*self.objs[1]

        self.batchsize = None

        if listt is not None:
            newObj = self.obj
            nindex = -1
            for index, model in enumerate(listt):
                obj = (1-self.lambdaa)*model.objs[0]+(self.lambdaa)*model.objs[1]
                if obj < newObj:
                    nindex = index
                    newObj = obj
            if nindex >= 0:
                self.theta = copy.deepcopy(listt[nindex].theta)
                self.obj = copy.deepcopy(listt[nindex].obj)
                self.objs = copy.deepcopy(listt[nindex].objs)

    # return the regularization parameter
    def getReg(self):
        return self.lambdaa

    # calculates the error for a rating
    def sampleError(self, rating, theta):
        u, i, score = rating
        uIndS = self.d*u
        uIndF = self.d*(u+1)
        iIndS = self.nU*self.d + self.d*i
        iIndF = self.nU*self.d + self.d*(i+1)
        uv = theta[uIndS:uIndF]
        iv = theta[iIndS:iIndF]
        return np.dot(uv, iv) - score

    # calculates the error objective
    def errorObj(self, theta):
        obj = 0
        for rating in self.rTrain:
            obj += self.sampleError(rating, theta)**2

        return obj/self.nR

    # calculates the regularization objective
    def regObj(self, theta):
        return np.dot(theta, theta)/theta.size

    # calculates the scalarized objective
    def fObj(self, theta):
        return (1-self.lambdaa)*self.errorObj(theta) \
            + self.lambdaa*self.regObj(theta)

    # calculates the gradient for the error objective
    def errorGrad(self, theta):
        grad = np.zeros(theta.shape)

        for rating in self.rTrain:
            u, i, score = rating
            uIndS = self.d*u
            uIndF = self.d*(u+1)
            iIndS = self.nU*self.d + self.d*i
            iIndF = self.nU*self.d + self.d*(i+1)
            error = self.sampleError(rating, theta)
            grad[uIndS:uIndF] += 2*theta[iIndS:iIndF]*error
            grad[iIndS:iIndF] += 2*theta[uIndS:uIndF]*error

        return grad/self.nR

    # calculates the gradient for the regularization objective
    def regGrad(self, theta):
        grad = 2*theta
        return grad/theta.size

    # calculates the gradient for the scalarized objective
    def fGrad(self, theta):
        return (1-self.lambdaa)*self.errorGrad(theta) \
            + self.lambdaa*self.regGrad(theta)

    # calculates the gradient for the error objective
    def errorHessD(self, theta, d):
        hessD = np.zeros(theta.shape)

        for rating in self.rTrain:
            u, i, score = rating
            uIndS = self.d*u
            uIndF = self.d*(u+1)
            iIndS = self.nU*self.d + self.d*i
            iIndF = self.nU*self.d + self.d*(i+1)
            hessD[uIndS:uIndF] += \
                2*theta[iIndS:iIndF]*np.dot(theta[iIndS:iIndF], d[uIndS:uIndF])
            hessD[iIndS:iIndF] +=  \
                2*theta[uIndS:uIndF]*np.dot(theta[uIndS:uIndF], d[iIndS:iIndF])

        return hessD/self.nR

    # calculates the gradient for the regularization objective
    def regHessD(self, theta, d):
        hessD = 2*d
        return hessD/theta.size

    # calculates the gradient for the scalarized objective
    def fHessD(self, theta, d):
        return (1-self.lambdaa)*self.errorHessD(theta, d) \
            + self.lambdaa*self.regHessD(theta, d)

    def batch(self, theta):
        if self.batchsize is not None:
            shuffle(self.ratings)
            self.rTrain = self.ratings[:int(self.batchsize*self.nR)]

    def optimize(self):
        t0 = time()
        out = opt.minimize(self.fObj, self.theta,
                           jac=self.fGrad, hessp=self.fHessD,
                           method='Newton-CG', options={'xtol': 1e-3})
        self.theta = out.x
        self.objs = [self.errorObj(self.theta), self.regObj(self.theta)]
        self.obj = self.fObj(self.theta)
        self.training_time = time() - t0
        self.users = self.theta[:self.nU*self.d].reshape((self.nU, self.d))
        self.items = self.theta[self.nU*self.d:].reshape((self.nI, self.d))
