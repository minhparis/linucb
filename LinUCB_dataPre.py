#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

LinUCB data preperation

Created on Wed Oct 23 14:10:58 2019
@author: DANG
"""
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# Data Preperation
class MovieLensData:
    """
MovieLens datas
MovieLens dataset prepared for LinUCB
"""

# Data Preperation
class MovieLensData:

    def __init__(self, d=30):
        '''d <int> is effectively the number of features'''
        self.n_users = 0
        self.n_movies = 0
        
        self.read_data()
        self.topN_movies_users()
        U, Vt, M = self.matrix_factorization(K=d)
        
        self.U = U
        self.Vt = Vt
        self.M = M / 5.
        
    def read_data(self):
        try:
            self.ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', 
                                    names = ['UserID', 'MovieID', 'Rating', 'Timestamp'],
                                    encoding = 'latin1',
                                    engine = 'python')
            self.movies  = pd.read_table('./ml-1m/movies.dat',  sep='::',
                                    names = ['MovieID', 'Title', 'Genres'], 
                                    encoding = 'latin1',
                                    engine ='python')
            self.users   = pd.read_table('./ml-1m/users.dat',  sep='::', 
                                    names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip'], 
                                    encoding = 'latin1',
                                    engine = 'python')
        except FileNotFoundError:
            print('Directory ml-1m not found. Download http://files.grouplens.org/datasets/movielens/ml-1m.zip and extract it to the working directory')
            raise

    def topN_movies_users(self, N = 1000):
        ratings_count = self.ratings.groupby(by='MovieID', as_index=True).size()
        top_ratings = ratings_count[ratings_count>=N]
        
        self.ratings_topN = self.ratings[self.ratings.MovieID.isin(top_ratings.index)]
        
        self.n_users = self.ratings_topN.UserID.unique().shape[0]
        self.n_movies = self.ratings_topN.MovieID.unique().shape[0]
    
    def matrix_factorization(self, save_data = False, K = 30):
        R_df = self.ratings_topN.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
        M = R_df.as_matrix()
        
        U, s, Vt = svds(M, k = K)
        s=np.diag(s)
        U = np.dot(U,s)
        if save_data:
            np.savetxt('U_movielens.csv', U, delimiter=',') 
            np.savetxt('Vt_movielens.csv', Vt, delimiter=',')
        return U, Vt, M
    
    def reward(self, user, item):
        return self.M[user, item]

    def active_users(self):
        return np.argwhere((self.M != 0).sum(1) > 150).flatten()
    
    def mean_rating(self, movie):
        non_nul = np.argwhere(self.M[:,movie] != 0).flatten()
        return np.mean(self.M[non_nul,movie])

#utility
def sparsity(arr):
        assert(arr.ndim==2)
        return round(1.0 - (np.count_nonzero(arr) / (arr.shape[0]*arr.shape[1])), 3)
