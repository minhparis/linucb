#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit UCB1 - Data Science Project
Group zambra

Created on Thu Oct 31 22:14:24 2019
@author: DANG
"""

# Normally alpha in this script should be 1
# But in that case it takes so many time to converge
# Another UCB1-Normal algorithme is presented with another way of calculating alpha
# See the link below
# http://homes.di.unimi.it/cesa-bianchi/Pubblicazioni/ml-02.pdf

import matplotlib.pyplot as plt
from LinUCB_dataPre import MovieLensData
import time
from collections import Counter
import numpy as np
import bandit_plotting

class LinUCB:
    def __init__(self, movielens_data, alpha=1,  delta=0):
        self.data = movielens_data
        self.alpha = alpha
        self.delta = delta
        
        self.n_movies = movielens_data.n_movies
        self.users = movielens_data.active_users()
        self.n_users = self.users.shape[0]
        
        self.d = self.data.Vt.shape[0]
        self.k = self.d
        
        self.k_mean_reward = np.zeros(self.n_movies)
        self.k_n = np.zeros(self.n_movies)
        self.n = 1

    def choose_arm(self, user):
        UCB = -1
        I = -1
        for i in range(self.n_movies):
            r = self.data.reward(self.users[user], i)
            if r == 0:
                continue
            if self.k_n[i] == 0:
                UCB = 0
                I= i
                break
            # print('this film has a score')
            UCB_i = self.k_mean_reward[i] + self.alpha*np.sqrt(2*np.log(self.n)/self.k_n[i])            
            if UCB_i > UCB:
                UCB = UCB_i
                I = i
        return I, self.k_mean_reward[I], UCB
    
    def fit(self, users, T = 50):
        regrets = []
        ratings = []
        films_rec = []
        ratings_taken_ucb = []
        ratings_taken_mean = []
        
        regret = 0
        total_reward = 0
        for i in range(T):
            for user in users:
                a,esti_mean, UCB = self.choose_arm(user)
                if a == -1:
                    print("we don't get a film recommendation for user {}".format(user))
                    continue
                
                r = self.data.reward(self.users[user], a)
                
                if r == 0:
                    print('this film has not a score')
                    continue
                r += np.random.normal(0,self.delta)
                
                            
                self.n += 1
                self.k_n[a] += 1
                
                total_reward += r
                self.k_mean_reward[a] = (self.k_mean_reward[a] * self.k_n[a] + r - self.k_mean_reward[a]) / self.k_n[a]
                                
                regret += 1. - r
                regrets.append(regret)
                ratings.append(r)
                if UCB > 0:
                    films_rec.append(a)
                ratings_taken_ucb.append(UCB)
                ratings_taken_mean.append(esti_mean)
                
        return regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb
    
def bandit_plot(regrets, ratings, films_rec, r_taken_mean, r_taken_ucb, data, lin_ucb, user):
    bandit_plotting.ratings(ratings)
    bandit_plotting.films_freq_rewards_2(films_rec, user, data, lin_ucb)
    
    bandit_plotting.plot_cum_regrets(regrets,"LinUCB", xsqrtlog=False)
    bandit_plotting.plot_cum_regrets(regrets,"LinUCB", xsqrtlog=True)
    
    bandit_plotting.rating_estimated(r_taken_mean, r_taken_ucb)

if 'movielens_data' not in locals():
    print('preparing data')
    movielens_data = MovieLensData()

niter = 3000
alpha = 0.3
delta = 0. # noise
lin_ucb = LinUCB(movielens_data, alpha, delta)

user = [0]

start = time.time()
regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb = lin_ucb.fit(user, niter)
end = time.time()
print("time used: {}".format(end - start))
bandit_plot(regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb, movielens_data, lin_ucb, user)
