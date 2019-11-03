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
from LinUCB_dataPre import MovieLensData
import matplotlib.pyplot as plt
import time

def plot_regret_com(regrets_dis, regrets_hyb):
    plt.plot(regrets_dis, label = 'regret cumulé - LinUCB disjoint')
    plt.plot(regrets_hyb, label = 'regret cumulé - LinUCB hybride')
    plt.title('regret cumulé comparaison')
    plt.legend()
    plt.show()

def plot_ratings_com(r_dis, r_hyb):
    plt.plot(r_dis, label = 'ratings - LinUCB disjoint')
    plt.plot(r_hyb, label = 'ratings - LinUCB hybride')
    plt.title('ratings comparaison')
    plt.legend()
    plt.show()

def plot_com(regrets_dis, regrets_hyb, r_dis, r_hyb):
    plot_regret_com(regrets_dis, regrets_hyb)
    plot_ratings_com(r_dis, r_hyb)
    
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
    
    plot_com(regrets_dis, regrets_hyb, r_dis, r_hyb)
    