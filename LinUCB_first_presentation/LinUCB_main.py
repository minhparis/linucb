#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit LinUCB - Data Science Project
Group zambra

LinUCB_main

Created on Wed Oct 23 19:00:05 2019
@author: DANG
"""
from LinUCB_dataPre import MovieLensData
from LinUCB_algo import LinUCB

import numpy as np
import matplotlib.pyplot as plt

import time


if __name__ == "__main__":
    print('preparing data')
    movielens_data = MovieLensData()
    
    
    niter = 30
    lin_ucb = LinUCB(movielens_data, context = 'user')
    
    # n_users = lin_ucb.n_users // 5
    n_users = 1
    l_users = list(range(n_users))
    l_users_train = np.random.choice(list(l_users), n_users, replace=False).tolist()
    l_users_test = list(set(l_users) - set(l_users_train))
    
    regrets_iter_train = []
    regrets_iter_test = []
    
    for i in range(niter):
      print("iter {}".format(i))
      start = time.time()
      regrets, ratings, film_rec_count = lin_ucb.fit(l_users_train)
      regrets_iter_train.append(np.mean(regrets))
      start1 = time.time()
      print("training time used: {}".format(start1 - start))
      regrets, ratings, film_rec_count = lin_ucb.evaluator(l_users_test)
      regrets_iter_test.append(np.mean(regrets))
      end = time.time()
      print("evaluate time used: {}".format(end - start1))
    
    plt.plot(regrets_iter_train)
    plt.plot(regrets_iter_test)
    plt.xlabel("nombre d'itérations")
    plt.ylabel("Regret moyen")
    plt.title("LinUCB Regret")
    plt.legend(['training set', 'test set'])
    plt.show()