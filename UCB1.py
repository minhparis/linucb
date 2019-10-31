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
                films_rec.append(a)
                ratings_taken_ucb.append(UCB)
                ratings_taken_mean.append(esti_mean)
                
        return regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb
    
def bandit_plot(regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb, movielens_data, lin_ucb, user):
    # films = Counter(films_rec)
    # films_keys = list(films.keys())
    
    # fig, ax = plt.subplots()  
    # ax.bar(np.arange(len(films.keys())), films.values())
    # plt.xticks(np.arange(len(films.keys())), films_keys)
    # for i, v in enumerate(films.values()):
    #     ax.text(i-0.2, v+0.1, str(int(movielens_data.M[lin_ucb.users[user],films_keys[i]]*5)), fontweight='bold')
    # plt.title("film recommendation frequence")
    # plt.show()
    
    
    plt.plot(ratings)
    plt.xlabel("T")
    plt.title("real rating")
    plt.show()
    
    # for movie in films_keys:
    #     plt.plot(ratings_ucb[:,movie],label="ucb rating {}".format(movie))
    #     plt.plot(ratings_esti_mean[:,movie], label="estimated mean rating {}".format(movie))
    # plt.xlabel("T")
    # plt.title("estimated rating of film 25")
    # plt.legend()
    # plt.show()
    
    plt.plot(ratings_taken_ucb,label="ucb rating")
    plt.plot(ratings_taken_mean, label="estimated mean rating")
    plt.xlabel("T")
    plt.title("estimated rating")
    plt.legend()
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
    plt.xlabel("sart(T)*log(T)")
    plt.ylabel("Regret cumulé")
    plt.title("LinUCB Regret cumulé")
    plt.show()

if 'movielens_data' not in locals():
    print('preparing data')
    movielens_data = MovieLensData()

niter = 3000
alpha = 0.2
delta = 0. # noise
lin_ucb = LinUCB(movielens_data, alpha, delta)

user = [0]

start = time.time()
regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb = lin_ucb.fit(user, niter)
end = time.time()
print("time used: {}".format(end - start))
bandit_plot(regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb, movielens_data, lin_ucb, user[0])
