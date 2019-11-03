#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

Created on Tue Oct 29 20:54:51 2019
@author: DANG
"""

import numpy as np
import time
from LinUCB_dataPre import MovieLensData
import bandit_plotting

class LinUCB_hybrid:
    def __init__(self, movielens_data, alpha=1, lambda_theta=1, lambda_beta=1, delta=0):
        self.data = movielens_data
        self.alpha = alpha
        self.delta = delta
        
        self.n_movies = movielens_data.n_movies
        self.users = movielens_data.active_users()
        self.n_users = self.users.shape[0]
        
        self.d = self.data.Vt.shape[0]
        self.k = self.d
        
        # self.x = movielens_data.Vt.transpose()
        self.x = np.zeros((self.n_movies, self.d))
        self.z = movielens_data.Vt.transpose()
        
        self.A = np.repeat(np.identity(self.d)[np.newaxis, :, :] * lambda_theta, self.n_movies, axis=0)
        self.b = np.repeat(np.zeros(self.d)[np.newaxis, :], self.n_movies, axis=0)
        self.B = np.repeat(np.zeros((self.d, self.k))[np.newaxis, :, :], self.n_movies, axis=0)
        self.A0 = np.identity(self.k)* lambda_beta
        self.b0 = np.zeros((self.k))
        
        
        self.recom_record = np.zeros((self.n_users, self.k))
        self.theta = np.zeros((self.n_movies, self.d))

        self.first_add = np.zeros((self.n_movies))

    def choose_arm(self, user):
        mean_t = np.zeros((self.n_movies))
        p_t = np.zeros((self.n_movies))
        A0_inv = np.linalg.inv(self.A0)
        
        self.theta = np.zeros((self.n_movies, self.d))
        self.beta = np.dot(A0_inv, self.b0)
        
        for a in range(self.n_movies):
            x = self.x[a,:].reshape(-1,1)
            z = self.z[a,:].reshape(-1,1)
            
            Aa_inv = np.linalg.inv(self.A[a])
            Ba = self.B[a]
            
            self.theta[a] = Aa_inv.dot(self.b[a] - Ba.dot(self.beta))
            s_ta = z.transpose().dot(A0_inv).dot(z) \
                    -2*z.transpose().dot(A0_inv).dot(Ba.transpose()).dot(Aa_inv).dot(x)\
                    +x.transpose().dot(Aa_inv).dot(x)\
                    +x.transpose().dot(Aa_inv).dot(Ba).dot(A0_inv).dot(Ba.transpose()).dot(Aa_inv).dot(x)
            if s_ta < 0:
                print("something's going wrong with {}th film".format(a))
                p_t[a] = 0
                continue
            mean_t[a] = z.transpose().dot(self.beta)+x.transpose().dot(self.theta[a,:])
            p_t[a] = mean_t[a] + self.alpha * np.sqrt(s_ta)
        
        #choose arm
        p_t_max = p_t.max()
        if p_t_max == 0:
            return -1,0,0
        best_arms = np.flatnonzero(p_t == p_t_max)
        a = np.random.choice(best_arms) # choose with ties broken
        return a, mean_t, p_t
    
    def fit(self, users, T = 50):
        regrets = []
        ratings = []
        films_rec = []
        ratings_ucb = []
        ratings_esti_mean = []
        ratings_taken_ucb = []
        ratings_taken_mean = []
        
        regret = 0
        for i in range(T):
            for user in users:
                a, mean_t, p_t = self.choose_arm(user)
                if a == -1:
                    print("we don't get a film recommendation for user {}".format(user))
                    continue
                
                x = self.x[a,:].reshape(-1,1)
                z = self.z[a,:].reshape(-1,1)
                r = self.data.reward(self.users[user], a) + np.random.normal(0,self.delta)
                
                if r == 0:
                    print('this film has not a score, using mean score')
                    r = self.data.mean_rating(a)
                
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
                ratings_esti_mean.append(mean_t)
                ratings_ucb.append(p_t)
                ratings_taken_mean.append(mean_t[a])
                ratings_taken_ucb.append(p_t[a])
            
        ratings_esti_mean = np.vstack(ratings_esti_mean)
        ratings_ucb = np.vstack(ratings_ucb)
        return regrets, ratings, films_rec, ratings_taken_mean, ratings_taken_ucb, ratings_esti_mean, ratings_ucb 
    
def bandit_plot(regrets, ratings, r_mean, r_ucb, r_taken_mean, r_taken_ucb, films_rec, data, lin_ucb, user):
    bandit_plotting.films_freq_rewards_2(films_rec, user, data, lin_ucb)
    bandit_plotting.ratings(ratings)
    
    bandit_plotting.ratings_estimated(films_rec, r_mean, flag = 'mean')
    bandit_plotting.ratings_estimated(films_rec, r_ucb, flag = 'ucb')
    
    bandit_plotting.rating_estimated(r_taken_mean, r_taken_ucb)
    
    bandit_plotting.plot_cum_regrets(regrets,"LinUCB", xsqrtlog=False)
    bandit_plotting.plot_cum_regrets(regrets,"LinUCB", xsqrtlog=True)


if __name__ == '__main__':
    if 'data' not in locals():
        print('preparing data')
        data = MovieLensData()
    
    niter = 500
    alpha = 1.6
    lambda_theta = 1.5
    lambda_beta = 1.0
    delta = 0. # noise
    lin_ucb = LinUCB_hybrid(data, alpha, lambda_theta, lambda_beta, delta)
    
    users = [0]
    
    start = time.time()
    regrets, ratings, films_rec, r_taken_mean, r_taken_ucb, r_esti_mean, r_ucb = lin_ucb.fit(users, niter)
    end = time.time()
    print("time used: {}".format(end - start))
    bandit_plot(regrets, ratings, r_esti_mean, r_ucb, r_taken_mean, r_taken_ucb, films_rec, data, lin_ucb, users)
