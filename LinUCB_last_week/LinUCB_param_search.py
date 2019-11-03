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

def plot_regret_comp(list_regrets, approach, param_name, list_param):
    for i in range(len(list_regrets)):
        plt.plot(list_regrets[i], label=param_name+" = "+str(list_param[i]))
    plt.title(approach + ' regret cumulé de différent '+param_name)
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
    alpha = 1.6
    delta = 0 # noise
    users = [0]

    #for linUCB disjoint
    lambda_s = [1., 1.5, 1.8]
    #for linUCB hybrid
    lambda_theta = 1.
    lambda_betas = [1.0, 1.2, 1.5]
    
    approach = 'hybrid'
    
    list_regrets = []
    if approach == 'disjoint':
        for lambda_ in lambda_s:
            start = time.time()
            lin_ucb = LinUCB_disjoint(data, alpha, lambda_, delta)    
            regrets_dis, r_dis, films_rec_dis, r_taken_mean_dis, r_taken_ucb_dis = lin_ucb.fit(users, niter)
            list_regrets.append(regrets_dis)
            end = time.time()
            print("LinUCB disjoint time used: {}".format(end - start))
        
    elif approach == 'hybrid': 
        for lambda_beta in lambda_betas:
            start = time.time()
            lin_ucb = LinUCB_hybrid(data, alpha, lambda_theta, lambda_beta, delta)
            regrets_hyb, r_hyb, films_rec_hyb, r_taken_mean_hyb, r_taken_ucb_hyb, r_esti_mean_hyb, r_ucb_hyb = lin_ucb.fit(users, niter)
            list_regrets.append(regrets_hyb)
            end = time.time()
            print("LinUCB hybride time used: {}".format(end - start))
    else:
        print('No corresponding approach')
        exit
    plot_regret_comp(list_regrets, approach, r'$\lambda$', lambda_betas)