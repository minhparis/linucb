#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

Created on Tue Oct 29 20:54:51 2019
@author: DANG
"""

# There's a bug calculating s_ta, in many cases it will be negative which is clearly wrong.
# still didn't figure out where the error come from

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
        
        # self.x = np.concatenate((movielens_data.Vt.transpose(), np.ones((self.n_movies,1))), axis = 1)
        # self.z = np.concatenate((movielens_data.Vt.transpose(), np.ones((self.n_movies,1))), axis = 1)
        self.x = movielens_data.Vt.transpose()
        self.z = movielens_data.Vt.transpose()
        
        self.d = self.x.shape[1]
        self.k = self.d
        
        self.A = np.repeat(np.identity(self.d)[np.newaxis, :, :] * lambda_, self.n_movies, axis=0)
        self.b = np.repeat(np.zeros(self.d)[np.newaxis, :], self.n_movies, axis=0)
        self.B = np.repeat(np.zeros((self.d, self.k))[np.newaxis, :, :] * lambda_, self.n_movies, axis=0)
        self.A0 = np.identity(self.k)
        self.b0 = np.zeros((self.k))
        
        
        self.recom_record = np.zeros((self.n_users, self.k))
        self.theta = np.zeros((self.n_movies, self.d))

        self.first_add = np.zeros((self.n_movies))

    def choose_arm(self, user):
        p_t = np.zeros((self.n_movies))
        A0_inv = np.linalg.inv(self.A0)
        
        self.theta = np.zeros((self.n_movies, self.d))
        self.beta = np.dot(A0_inv, self.b0)
        
        for a in range(self.n_movies):
            x = self.x[a,:].reshape(-1,1)
            z = self.z[a,:].reshape(-1,1)
            
            r = self.data.reward(self.users[user], a)
            if r == 0:
                p_t[a] = -1e2
                continue
            
            Aa_inv = np.linalg.inv(self.A[a])
            Ba = self.B[a]
            
            self.theta[a] = np.dot(Aa_inv, self.b[a])
            s_ta = z.transpose().dot(A0_inv).dot(z)
            s_ta -= 2*z.transpose().dot(A0_inv).dot(Ba.transpose()).dot(Aa_inv).dot(x)
            s_ta += x.transpose().dot(Aa_inv).dot(x)
            s_ta += x.transpose().dot(Aa_inv).dot(Ba).dot(A0_inv).dot(Ba.transpose()).dot(Aa_inv).dot(x)
            if s_ta < 0:
                print("damn it something's going wrong with {}th film".format(a))
                continue
            p_t[a] = z.transpose().dot(self.beta)+x.transpose().dot(self.theta[a,:]) + self.alpha * np.sqrt(s_ta)
        
        #choose arm
        best_arms = np.flatnonzero(p_t == p_t.max())
        a = np.random.choice(best_arms) # choose with ties broken
        return a
    
    def fit(self, users, T = 50):
        regrets = []
        ratings = []
        films_rec = []
        
        regret = 0
        for i in range(T):
            for user in users:
                a = self.choose_arm(user)
                
                x = self.x[a,:].reshape(-1,1)
                z = self.z[a,:].reshape(-1,1)
                r = self.data.reward(self.users[user], a) + np.random.normal(0,self.delta)
                
                if r == 0:
                    print('this film has not a score')
                    continue
                
                Aa_inv = np.linalg.inv(self.A[a])
                
                self.A0 += self.B[a].transpose().dot(Aa_inv).dot(self.B[a])
                self.b0 += self.B[a].transpose().dot(Aa_inv).dot(self.b[a])
                self.A[a] += x.dot(x.transpose())
                self.B[a] += x.dot(z.transpose())
                self.b[a] += (r * x).flatten()
                self.A0 += z.dot(z.transpose()) - self.B[a].transpose().dot(Aa_inv).dot(self.B[a])
                self.b0 += r * z.flatten() - self.B[a].transpose().dot(Aa_inv).dot(self.b[a])
                
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
alpha = 1
lambda_ = 1
delta = 0. # noise
lin_ucb = LinUCB(movielens_data, alpha, lambda_, delta)

user = [0]

start = time.time()
regrets, ratings, films_rec = lin_ucb.fit(user, niter)
end = time.time()
print("time used: {}".format(end - start))
bandit_plot(regrets, ratings, films_rec)
