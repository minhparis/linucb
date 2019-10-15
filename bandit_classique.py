# -*- coding: utf-8 -*-
"""
Data Science Project

Group zambra

Bandit classique avec la politique random et la politique epsilon greedy
"""

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k, T, politique, epsilon = 0):
        self.k = k #number of arms
        self.T = T # total step
        self.politique = politique
        self.epsilon = epsilon
        
        self.n = 0 #step count
        self.k_n = np.zeros(k) #step count of each arm
        
        self.mu = np.random.uniform(0,1,k) # generate reward of each arm
        self.max_mu = max(self.mu) # maximum reward
        
        self.k_mean_reward = np.zeros(k)
        self.mean_reward = np.zeros(T)
        
        self.regret = np.zeros(T)
        self.mean_regret = np.zeros(T)
    
    def run(self):
        total_reward = 0
        total_regret = 0
        for i in range(self.T):
            if self.politique == 'random':
                I = np.random.choice(self.k)
            else:
                tmp = np.random.rand()
                if tmp < self.epsilon:
                    I = np.random.choice(self.k)
                else:
                    I = np.argmax(self.k_mean_reward)
                    
            reward =  self.mu[I]
            
            
            self.n += 1
            self.k_n[I] += 1
            
            total_reward += reward
            self.mean_reward[i] += total_reward/ self.n
            self.k_mean_reward[I] += (reward-self.k_mean_reward[I])/self.k_n[I]
            
            total_regret += self.max_mu - reward
            self.mean_regret[i] = total_regret / self.n
            self.regret[i] = total_regret
    
    
if __name__ == "__main__":
    # parameters
    k =10
    T =10000 #number of step
    niter = 10 #iterate 10 times to get average values
    politique = 'greedy'
    espsilon = 0.005
    
    # vector definition
    regret = np.zeros((T))
    mean_regret = np.zeros((T))
    reward = np.zeros((T))
    
    # bandit
    for i in range(niter):
        bandit = Bandit(k,T,politique, espsilon)
        bandit.run()
        regret = bandit.regret
        mean_regret += bandit.mean_regret
        reward += bandit.mean_reward
    regret /= niter
    mean_regret /= niter
    reward /= niter
    
    # plot
    plt.plot(reward)
    plt.title('mean reward')
    plt.show()
    plt.plot(mean_regret)
    plt.title('mean regret')
    plt.show()
    plt.plot(regret)
    plt.title('regret')
    plt.show()