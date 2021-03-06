#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

Created on Sat Nov  2 23:49:22 2019
@author: DANG
"""

from LinUCB_disjoint import LinUCB_disjoint
from LinUCB_hybride import LinUCB_hybrid
from UCB1 import LinUCB1
from LinUCB_dataPre import MovieLensData
import matplotlib.pyplot as plt
import time

def plot_regret_comp(regrets_dis, regrets_hyb, regrets_ucb1):
    plt.plot(regrets_ucb1, label = 'UCB1')
    plt.plot(regrets_dis, label = 'LinUCB disjoint')
    plt.plot(regrets_hyb, label = 'LinUCB hybride')
    plt.title('regret cumulé comparaison')
    plt.xlabel('T')
    plt.ylabel('regert')
    plt.legend()
    plt.show()

def plot_ratings_comp(r_dis, r_hyb, r_ucb1):
    plt.plot(r_ucb1, label = 'ratings - UCB1')
    plt.plot(r_dis, label = 'ratings - LinUCB disjoint')
    plt.plot(r_hyb, label = 'ratings - LinUCB hybride')
    plt.title('ratings comparaison')
    plt.legend()
    plt.show()

def plot_comp(regrets_dis, regrets_hyb, regrets_ucb1, r_dis, r_hyb, r_ucb1):
    plot_regret_comp(regrets_dis, regrets_hyb, regrets_ucb1)
    plot_ratings_comp(r_dis, r_hyb, r_ucb1)
    
if __name__ == '__main__':
    if 'data' not in locals():
        print('preparing data')
        data = MovieLensData()
    niter = 500
    alpha = 1.6
    lambda_ = 1
    delta = 0. # noise
    
    users = [0]
    
    start = time.time()
    lin_ucb = LinUCB_disjoint(data, alpha, lambda_, delta)    
    regrets_dis, r_dis, films_rec_dis, r_taken_mean_dis, r_taken_ucb_dis = lin_ucb.fit(users, niter)
    end = time.time()
    print("LinUCB disjoint time used: {}".format(end - start))
    
    lambda_theta = 1.0
    lambda_beta = 1.0
    delta = 0. # noise
        
    start = time.time()
    lin_ucb = LinUCB_hybrid(data, alpha, lambda_theta, lambda_beta, delta)
    regrets_hyb, r_hyb, films_rec_hyb, r_taken_mean_hyb, r_taken_ucb_hyb, r_esti_mean_hyb, r_ucb_hyb = lin_ucb.fit(users, niter)
    end = time.time()
    print("LinUCB hybride time used: {}".format(end - start))
    
    
    start = time.time()
    lin_ucb = LinUCB1(data, alpha, delta)
    regrets_ucb1, r_ucb1, films_rec, ratings_taken_mean, ratings_taken_ucb = lin_ucb.fit(users, niter)
    end = time.time()
    print("UCB1 time used: {}".format(end - start))
    
    plot_comp(regrets_dis, regrets_hyb, regrets_ucb1, r_dis, r_hyb, r_ucb1)
    