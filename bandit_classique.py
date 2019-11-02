# -*- coding: utf-8 -*-
"""
Bandit classique - Data Science Project
Group zambra

Bandit classique
Created on Sat Oct 19 21:55:42 2019

@author: DANG
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class Bandit:
    def __init__(self, k, T):
        self.k = k #number of arms
        self.T = T # total step
        
        self.n = 0 #step count
        self.k_n = np.zeros(k) #step count of each arm
        
        self.mu = np.random.uniform(0,1,k) # generate reward of each arm
        self.max_mu = max(self.mu) # maximum reward
        
        self.k_mean_reward = np.zeros(k)
        self.mean_reward = np.zeros(T)
        
        self.regret = np.zeros(T)
        self.mean_regret = np.zeros(T)
        self.choices = np.zeros(T)

    def get_reward(self, I):
        # reward = self.mu[I]
        reward = np.random.normal(self.mu[I],1,1)
        return reward
    
    def argmax_UCB(self):
        UCB = -1
        I = -1 
        for i in range(self.k):
            UCB_i = self.k_mean_reward[i] + math.sqrt(2*math.log(self.n)/self.k_n[i])
            if UCB_i > UCB:
                UCB = UCB_i
                I = i
        return I
    
    def get_I_greedy(self, epsilon):
        tmp = np.random.rand()
        if tmp < (epsilon / math.sqrt(self.n+1)):
            I = np.random.choice(self.k)
        else:
            I = np.argmax(self.k_mean_reward)
        return I
    
    def get_I_UCB(self, i):
        if i < self.k:
            I = i
        else:
            I = self.argmax_UCB()
        return I
    
    def run(self, politique, epsilon = 0.005):
        total_reward = 0
        total_regret = 0
        for i in range(self.T):
            if politique == 'random':
                I = np.random.choice(self.k)
            elif politique == 'greedy':
                I = self.get_I_greedy(epsilon)
            else:
                I = self.get_I_UCB(i)
                    
            self.choices[i] = I
            reward =  self.get_reward(I)
            
            self.n += 1
            self.k_n[I] += 1
            
            total_reward += reward
            self.mean_reward[i] += total_reward/ self.n
            # self.k_mean_reward[I] += (reward-self.k_mean_reward[I])/self.k_n[I]
            self.k_mean_reward[I] = (self.k_mean_reward[I] * self.k_n[I] + reward - self.k_mean_reward[I]) / self.k_n[I]
            
            # total_regret += self.max_mu - reward
            total_regret += self.max_mu - self.mu[I]
            self.mean_regret[i] = total_regret / self.n
            self.regret[i] = total_regret
      

def run_bandit(k, T, niter=10, politique = 'UCB', epsilon=0.005):
    # vector definition
    regret = np.zeros((T))
    mean_regret = np.zeros((T))
    mean_reward = np.zeros((T))
    
    # bandit
    for i in range(niter):
        bandit = Bandit(k,T)
        bandit.run(politique, epsilon)
        regret = bandit.regret
        mean_regret += bandit.mean_regret
        mean_reward += bandit.mean_reward
    regret /= niter
    mean_regret /= niter
    mean_reward /= niter
    return bandit, mean_reward, regret, mean_regret


def bandit_plot(bandit, mean_reward, regret, mean_regret, flag):
    plt.plot(mean_reward)
    plt.title('mean reward')
    plt.show()
    plt.plot(mean_regret)
    plt.title('mean regret')
    plt.show()
    plt.plot(bandit.choices)
    plt.title('choices')
    plt.show()
    plt.hist(bandit.choices, bins = 10)
    plt.show()
    if flag == 'loglog':
        plt.loglog(regret)
        plt.title('regret')
        plt.show()
    elif flag == 'log':
        plt.xscale('log')
        plt.plot(regret)
        plt.title('regret')
        plt.show()
    else:
        plt.plot(regret)
        plt.title('regret')
        plt.show()
        
def plot_regret_comp(regret_random, regret_greedy, regret_ucb):
    plt.plot(regret_random, label='random')
    plt.plot(regret_greedy, label='$epsilon$-greedy')
    plt.plot(regret_ucb, label='UCB1')
    plt.title('regret cumulé')
    plt.legend()
    plt.show()

def single_test(politique, k, T, niter, epsilon):
    
    if politique == 'greedy':
        flag = 'loglog' 
    elif politique == 'UCB':
        flag = 'normal'
    else:
        flag = 'normal'
    bandit, mean_reward, regret, mean_regret = run_bandit(k, T, niter, politique, epsilon)
    bandit_plot(bandit, mean_reward, regret, mean_regret, flag)

def multi_test(k, T, niter, epsilon):
    bandit_random, mean_reward_random, regret_random, mean_regret_random = run_bandit(k, T, niter, 'random', epsilon)
    bandit_greedy, mean_reward_greedy, regret_greedy, mean_regret_greedy = run_bandit(k, T, niter, 'greedy', epsilon)
    bandit_ucb, mean_reward_ucb, regret_ucb, mean_regret_ucb = run_bandit(k, T, niter, 'ucb', epsilon)
    plot_regret_comp(regret_random, regret_greedy, regret_ucb)

if __name__ == "__main__":
    # parameters
    k =10
    T =10000 #number of step
    niter = 1 #iterate 10 times to get average values
    politique = 'greedy' # random, greedy or UCB
    epsilon = 5
    
    multi_test(k, T, niter, epsilon)