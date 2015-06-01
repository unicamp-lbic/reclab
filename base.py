# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:23:23 2015

@author: thalita
"""
import abc
from heapq import heappush, heappop
import numpy as np

class BaseDatabase(object):
    __metaclass__ = abc.ABCMeta

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, val):
        self._matrix = val

    @abc.abstractmethod
    def get_matrix(self):
        "return data matrix"
        return self.matrix

    @abc.abstractmethod
    def n_users(self):
        return self.matrix.shape[0]

    @abc.abstractmethod
    def get_rating(self, user_id, item_id):
        return self.matrix[user_id, item_id]

    @abc.abstractmethod
    def get_user_vector(self, user_id):
        "return a 2D array with user ratings"
        return self.matrix[user_id, :]

    @abc.abstractmethod
    def get_item_vector(self, item_id):
        "return a 2D array with item ratings"
        return self.matrix[:,item_id]

    @abc.abstractmethod
    def get_unrated_items(self, user_id):
        "return unrated item ids for user"
        return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                if rating == 0]

    @abc.abstractmethod
    def get_rated_items(self, user_id):
        "return items rated by user_id user"
        return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                if rating > 0]

    @abc.abstractmethod
    def get_rated_users(self, item_id):
        "return users who did not rate item_id user"
        return [idx for idx, rating in enumerate(self.matrix[:, item_id])
                if rating > 0]



class BaseRecommender(object):
    __metaclass__ = abc.ABCMeta

    @property
    def database(self):
        "Database object"
        return self._database

    @database.setter
    def database(self, val):
        self._database = val

    @abc.abstractmethod
    def recommend(self, target_user, topN):
        "return recomendation list for target_user"
        return


class RatingPredictor(BaseRecommender):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self):
        "learn recommender model (neighborhood, matrix factorization, etc)"
        return self

    @abc.abstractmethod
    def predict(self, target_user, target_item):
        return

    def recommend(self, target_user, how_many=np.inf, threshold=0,
                  candidate_items = None):
        unrated = self.database.get_unrated_items(target_user) \
                  if candidate_items is None else candidate_items
        ratings = []
        for item in unrated:
            # add tuples (-rating, item) to min heap
            pred_rating = self.predict(target_user, item)
            if pred_rating > threshold:
                heappush(ratings, (-pred_rating, item))
        if ratings is []:
            print('No recomendation could be made for this user')
            print(ratings)
        lenght = min(how_many, len(ratings))
        rec_list = []
        for _ in range(lenght):
            # pop tuple (item_id, rating) from ratings heap
            # and push into rec_list
            pred_rating, item = heappop(ratings)
            rec_list.append((item, -pred_rating))
        return rec_list


class BaseEnsemble(BaseRecommender):
    __metaclass__ = abc.ABCMeta

    @property
    def RS_list(self):
        "list of recommenders used by the ensemble"
        return self._RS_list

    @RS_list.setter
    def RS_list(self, val):
        self._RS_list = val

    @abc.abstractmethod
    def fit(self, database, **varargs):
        "learn recommender model (neighborhood, matrix factorization, etc)"
        self.database = database
        for RS in self.RS_list:
            RS.fit(self.database, **varargs)
        return self