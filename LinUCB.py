#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

Created on Mon Oct 28 21:56:53 2019
@author: DANG
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from LinUCB_dataPre import MovieLensData
import time
from collections import Counter
import numpy as np

class LinUCB:
    def __init__(self, movielens_data, alpha = 1, lambda_ = 1, delta = 0):
        self.data = movielens_data
        self.alpha = alpha
        self.delta = delta
        
        self.n_movies = movielens_data.n_movies
        self.users = movielens_data.active_users()
        self.n_users = self.users.shape[0]
        
        self.X = np.concatenate((movielens_data.Vt.transpose(), np.ones((self.n_movies,1))), axis = 1)
        self.X = normalize(self.X, axis=0)

        self.d = self.X.shape[1]
        
        self.A = np.repeat(np.identity(self.d)[np.newaxis, :, :] * lambda_, self.n_movies, axis=0)
        self.b = np.repeat(np.zeros(self.d)[np.newaxis, :], self.n_movies, axis=0)
        
        self.recom_record = np.zeros((self.n_users, self.n_movies))
        self.theta = np.zeros((self.n_movies, self.d))

        self.first_add = np.zeros((self.n_movies))

    def choose_arm(self, user):
        p_t = np.zeros((self.n_movies))
        self.theta = np.zeros((self.n_movies, self.d))
        
        for a in range(self.n_movies):
            x = self.X[a,:].reshape(-1,1)
            r = self.data.reward(self.users[user], a)
            if r == 0:
                p_t[a] = -1e2
                continue
            A_a_inv = np.linalg.inv(self.A[a])
            self.theta[a,:] = np.dot(A_a_inv, self.b[a].transpose()).transpose()
            p_t[a] = np.dot(self.theta[a,:], x) + self.alpha * np.sqrt(np.dot(np.dot(x.transpose(), A_a_inv), x))
        #choose arm
        best_arms = np.flatnonzero(p_t == p_t.max())
        # print('pta: \n{}'.format(p_t))
        # print('max: {}, \nfilm: \n{}'.format(p_t.max(), best_arms))
        a = np.random.choice(best_arms) # choose with ties broken
        return a
    
    def fit(self, user, T = 50):
        regrets = []
        ratings = []
        films_rec = []
        
        regret = 0
        for i in range(T):
            a = self.choose_arm(user)
            
            x = self.X[a,:].reshape(-1,1)
            r = self.data.reward(self.users[user], a) + np.random.normal(0,self.delta)
            
            if r == 0:
                print('this film has not a score')
                continue
            
            self.A[a] += np.dot(x, x.transpose())
            self.b[a] += (r * x).flatten()
            
            regret += 1. - r
            regrets.append(regret)
            ratings.append(r)
            films_rec.append(a)
            
        return regrets, ratings, films_rec
    
def bandit_plot(regrets, ratings, films_rec):
    films = Counter(films_rec)
    plt.bar(np.arange(len(films.keys())), films.values())
    plt.xticks(np.arange(len(films.keys())), films.keys())
    plt.title("film recommendation frequence")
    plt.show()
    
    plt.plot(ratings)
    plt.xlabel("T")
    plt.title("rating")
    plt.show()
    
    plt.figure(figsize=(13,6))
    plt.subplot(121)
    plt.plot(regrets)
    plt.xlabel("T")
    plt.ylabel("Regret cumulé")
    plt.title("LinUCB Regret cumulé")
    
    plt.subplot(122)
    xs = [np.sqrt(i)*np.log(i) for i in range(1,len(regrets)+1)]
    plt.plot(xs, regrets)
    plt.xlabel("sqrt(T)*log(T)")
    plt.ylabel("Regret cumulé")
    plt.title("LinUCB Regret cumulé")
    plt.show()

if 'movielens_data' not in locals():
    print('preparing data')
    movielens_data = MovieLensData()

niter = 500
alpha = 2.8
lambda_ = 2
delta = 0. # noise
lin_ucb = LinUCB(movielens_data, alpha, lambda_, delta)

user = 0

start = time.time()
regrets, ratings, films_rec = lin_ucb.fit(user, niter)
end = time.time()
print("time used: {}".format(end - start))
bandit_plot(regrets, ratings, films_rec)
