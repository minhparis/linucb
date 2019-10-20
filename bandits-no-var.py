import numpy as np
import matplotlib.pyplot as plt
import ipdb

def uniform_draw(K):
    '''random policy'''
    return np.random.randint(K) #uniform draw from [0,K-1]

def eps_greedy(K, eps, known_best_arm):
    '''epsilon-greedy policy'''
    if np.random.rand() < eps: #explore w/prob eps
        return np.random.randint(K) #uniform draw from [0,K-1]
    else: #exploit w/prob 1-eps
        return known_best_arm

def run(policy, K=10, T=5000, no_variance=True, return_all=False):
    ''' Main routine running stochastic MAB with a given policy
    args:
        policy -  is a str: "uniform", "epsgreedy", "ucb"
        K - no of arms indexed 0...K-1
        T - Total no of trials
        no_variance - boolean True if we ignore var
        return_all - boolean returns only the total regret at T if False
    '''
    #create all arrays populated with -1
    arm = np.full(T, -1) #actions vector
    known_best_arm = np.full(T, -1)
    reward = np.full(T, -1).astype(np.float64) #rewards vector cosequence of action
    total_reward = np.full(T, -1).astype(np.float64)
    total_regret = np.full(T, -1).astype(np.float64)

    #generate arms' distributions X_{i,t}
    if no_variance == True:
        # same for all t as there is no variance -> arms identified by constant means \mu_i
        arms_means = np.random.rand(K) # bounds [0,1) for rewards
    else:
        raise NotImplementedError
    
    #identify best arm \mu^{\ast} i.e. distribution with the highest mean
    mean_best_ind, mean_best = np.argmax(arms_means), np.max(arms_means)

    #initial iteration t=0
    arm[0] = np.random.randint(K) #first neccessary exploration
    reward[0] = arms_means[arm[0]] #later on add variance and draw form a distrbution here
    known_best_arm[0] = arm[0] #ignorance is bliss
    total_reward[0] = reward[0]
    total_regret[0] = mean_best - reward[0] #initial one!
    for t in range(1,T):
        #ipdb.set_trace()
        # select an arm that is pulled at t
        if policy == "uniform":
            arm[t] = uniform_draw(K) 
        elif policy == "epsgreedy":
            arm[t] = eps_greedy(K, .03, known_best_arm[t-1])
        elif policy == "ucb":
            raise NotImplementedError
        assert(arm[t] != -1) #make sure an arm was selected
        
        reward[t] = arms_means[arm[t]] # without variance, reward is the exact mean of an arm
        if reward[t] > arms_means[known_best_arm[t-1]]: #update \hat{\mu} ?
            known_best_arm[t] = arm[t]
        else:
            known_best_arm[t] = known_best_arm[t-1] #replace a -1 placeholder with a previous arm
        total_reward[t] = np.sum(reward[:t+1])
        total_regret[t] = t*mean_best - np.sum(reward[:t+1])
        #make all returned arrays internals later
    if return_all == True:
        return arm,arms_means,reward,known_best_arm,total_reward,total_regret 
    else:
        return np.sum(total_regret)
arm, arms_means, reward, known_best_arm, total_reward, total_regret = run("epsgreedy", K=10, T=5000, no_variance=True, return_all=True)

plt.subplots_adjust(hspace=.4)
plt.subplot(2, 2, 1)
plt.plot(range(len(arm)), arm)
plt.title('Arm')
plt.xlabel('t')
plt.ylabel('i')

plt.subplot(2, 2, 2)
plt.plot(range(len(known_best_arm)), known_best_arm)
plt.title('Known best arm')
plt.xlabel('t')
plt.ylabel('i')

plt.subplot(2, 2, 3)
plt.plot(range(len(reward)), reward)
plt.title('Reward')
plt.xlabel('t')
plt.ylabel('r')

plt.subplot(2, 2, 4)
plt.plot(range(len(total_regret)), total_regret)
plt.title('Total Regret')
plt.xlabel('t')
plt.ylabel('R')

plt.show()

mean_best_ind, mean_best = np.argmax(arms_means), np.max(arms_means)
print(mean_best_ind, mean_best)
print(np.unique(arm))
print("Total regret = {}".format(np.sum(total_regret)))
