#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

Created on Sun Nov  3 12:09:56 2019
@author: DANG
"""

from LinUCB_disjoint import LinUCB_disjoint
from LinUCB_hybride import LinUCB_hybrid
from LinUCB_dataPre import MovieLensData
import matplotlib.pyplot as plt
import time

def plot_regret_comp(list_regrets, param_name, list_param):
    for i in range(len(list_regrets)):
        plt.plot(list_regrets[i], label=param_name+" = "+str(list_param[i]))
    plt.title('regret cumulé de différent '+param_name)
    plt.xlabel('T')
    plt.ylabel('regret')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    '''find best parameters for linUCB'''
    
    if 'data' not in locals():
        print('preparing data')
        data = MovieLensData()
        
    niter = 500
    alphas = [1.2, 1.4, 1.6]
    delta = 0. # noise
    users = [0]

    #for linUCB disjoint
    lambda_ = 1
    #for linUCB hybrid
    lambda_theta = 1.0
    lambda_beta = 1.0
    
    approach = 'hybrid'
    
    list_regrets = []
    if approach == 'disjoint':
        for alpha in alphas:
            start = time.time()
            lin_ucb = LinUCB_disjoint(data, alpha, lambda_, delta)    
            regrets_dis, r_dis, films_rec_dis, r_taken_mean_dis, r_taken_ucb_dis = lin_ucb.fit(users, niter)
            list_regrets.append(regrets_dis)
            end = time.time()
            print("LinUCB disjoint time used: {}".format(end - start))
        
    elif approach == 'hybrid': 
        for alpha in alphas:
            start = time.time()
            lin_ucb = LinUCB_hybrid(data, alpha, lambda_theta, lambda_beta, delta)
            regrets_hyb, r_hyb, films_rec_hyb, r_taken_mean_hyb, r_taken_ucb_hyb, r_esti_mean_hyb, r_ucb_hyb = lin_ucb.fit(users, niter)
            list_regrets.append(regrets_hyb)
            end = time.time()
            print("LinUCB hybride time used: {}".format(end - start))
    
    plot_regret_comp(list_regrets, r'$\alpha$', alphas)