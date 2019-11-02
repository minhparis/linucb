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
import numpy as np
import bandit_plotting

class LinUCB_disjoint:
    def __init__(self, movielens_data, alpha = 1, lambda_ = 1, delta = 0):
        self.data = movielens_data
        self.alpha = alpha
        self.delta = delta
        
        self.n_movies = movielens_data.n_movies
        self.users = movielens_data.active_users()
        self.n_users = self.users.shape[0]
        
        # append constant 1 to every movie vector
        #self.X = np.concatenate((movielens_data.Vt.transpose(), np.ones((self.n_movies,1))), axis = 1)
        self.X = movielens_data.Vt.transpose()
        self.d = self.X.shape[1]
        
        self.A = np.repeat(np.identity(self.d)[np.newaxis, :, :] * lambda_, self.n_movies, axis=0)
        self.b = np.repeat(np.zeros(self.d)[np.newaxis, :], self.n_movies, axis=0)
        
        self.recom_record = np.zeros((self.n_users, self.n_movies))
        self.theta = np.zeros((self.n_movies, self.d))

        self.first_add = np.zeros((self.n_movies))

    def choose_arm(self, user):
        p_t = np.zeros((self.n_movies))
        p_t_mean = np.zeros((self.n_movies))
        self.theta = np.zeros((self.n_movies, self.d))
        
        for a in range(self.n_movies):
            x = self.X[a,:].reshape(-1,1)
            # r = self.data.reward(self.users[user], a)
            # if r == 0:
            #     p_t[a] = -1e2
            #     continue
            A_a_inv = np.linalg.inv(self.A[a])
            self.theta[a,:] = np.dot(A_a_inv, self.b[a].transpose()).transpose()
            p_t_mean[a] = np.dot(self.theta[a,:], x)
            p_t[a] = p_t_mean[a] + self.alpha * np.sqrt(x.transpose().dot(A_a_inv).dot(x))
        #choose arm
        best_arms = np.flatnonzero(p_t == p_t.max())
        # print('pta: \n{}'.format(p_t))
        # print('max: {}, \nfilm: \n{}'.format(p_t.max(), best_arms))
        a = np.random.choice(best_arms) # choose with ties broken
        return a, p_t_mean[a], p_t[a]
    
    def fit(self, users, T = 50):
        regrets = []
        ratings = []
        films_rec = []
        ratings_taken_mean = [] #mean score of chosen movies
        ratings_taken_ucb = [] #ucb score of chosen movies
        
        regret = 0
        for i in range(T):
            for user in users:
                a, p_t_mean, p_ta = self.choose_arm(user)
                
                x = self.X[a,:].reshape(-1,1)
                r = self.data.reward(self.users[user], a) + np.random.normal(0,self.delta)
                
                if r == 0:
                    print('this film has not a score, using mean score')
                    r = self.data.mean_rating(a)
                
                self.A[a] += np.dot(x, x.transpose())
                self.b[a] += (r * x).flatten()
                
                regret += 1. - r
                regrets.append(regret)
                ratings.append(r)
                films_rec.append(a)
                ratings_taken_mean.append(p_t_mean)
                ratings_taken_ucb.append(p_ta)
            
        return regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb
    
def bandit_plot(regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb, all_films_rewards):
    bandit_plotting.plot_cum_regrets(regrets,"LinUCB", xsqrtlog=False)
    bandit_plotting.plot_cum_regrets(regrets,"LinUCB", xsqrtlog=True)
    bandit_plotting.films_freq_rewards(films_rec, all_films_rewards)
    # bandit_plotting.all_films_rewards(all_films_rewards)
    bandit_plotting.ratings(ratings)
    bandit_plotting.rating_estimated(ratings_taken_mean, ratings_taken_ucb)
    

if __name__ == '__main__':
    if 'data' not in locals():
        print('preparing data')
        data = MovieLensData()
        
    niter = 500
    alpha = 1.7
    lambda_ = 1
    delta = 0. # noise
    lin_ucb = LinUCB_disjoint(data, alpha, lambda_, delta)
    
    users = [0]
    
    start = time.time()
    regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb = lin_ucb.fit(users, niter)
    end = time.time()
    print("time used: {}".format(end - start))
    
    all_films_rewards = lin_ucb.data.reward(lin_ucb.users[users],np.arange(lin_ucb.n_movies))
    bandit_plot(regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb, all_films_rewards[0])
