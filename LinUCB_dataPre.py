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
def read_data():
    ratings = pd.read_table('ml-1m/ratings.dat', sep='::', 
                            names = ['UserID', 'MovieID', 'Rating', 'Timestamp'],
                            encoding = 'latin1',
                            engine = 'python')
    movies  = pd.read_table('ml-1m/movies.dat',  sep='::',
                            names = ['MovieID', 'Title', 'Genres'], 
                            encoding = 'latin1',
                            engine ='python')
    users   = pd.read_table('ml-1m/users.dat',  sep='::', 
                            names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip'], 
                            encoding = 'latin1',
                            engine = 'python')
    return ratings, movies, users

def topN_movies_users(ratings, N = 1000):
    ratings_count = ratings.groupby(by='MovieID', as_index=True).size()
    top_ratings = ratings_count[ratings_count>=N]
    
    ratings_topN = ratings[ratings.MovieID.isin(top_ratings.index)]
    
    n_users = ratings_topN.UserID.unique().shape[0]
    n_movies = ratings_topN.MovieID.unique().shape[0]
    return ratings_topN, n_users, n_movies

def matrix_factorization(ratings_topN, save_data, K = 30):
    R_df = ratings_topN.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    M = R_df.as_matrix()
    
    U, s, Vt = svds(M, k = K)
    s=np.diag(s)
    U = np.dot(U,s)
    if save_data:
        np.savetxt('U_movielens.csv', U, delimiter=',') 
        np.savetxt('Vt_movielens.csv', Vt, delimiter=',')
    return U, Vt
    
def get_data(save_data = False):
    ratings, movies, users = read_data()
    ratings_topN, n_users, n_movies = topN_movies_users(ratings)
    U, Vt = matrix_factorization(ratings_topN, save_data)
    return U, Vt