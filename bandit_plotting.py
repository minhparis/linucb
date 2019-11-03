#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.ticker import FormatStrFormatter
plt.style.use('seaborn')

def plot_cum_regrets(regrets, algo_name, xsqrtlog=False):
    '''regrets <list>, algo_name <str>, xsqrtlog <bool>, show <bool>'''
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.set_title("{} Cumulative Regret".format(algo_name))
    ax.set_ylabel("$R(t)$")
    if xsqrtlog:
        x = [np.sqrt(i)*np.log(i) for i in range(1,len(regrets)+1)]
        ax.set_xlabel("$\\sqrt{t}\\log(t)$")
        savepath = 'cum_regrets_xsqrtlog_{}.png'.format(algo_name)
    else:
        x = range(len(regrets))
        ax.set_xlabel("$t$")
        savepath = 'cum_regrets_{}.png'.format(algo_name)
    ax.plot(x,regrets,antialiased=True)
    fig.savefig(savepath, format='png', bbox_inches='tight', dpi=400)

def films_freq_rewards(films_rec, all_films_rewards):
    '''films_rec <list> of pulled arm indices
    all_films_rewards <list> of rewards
    '''

    films = Counter(films_rec)
    sorted_films_counts = films.most_common()
    films_ind = [i[0] for i in sorted_films_counts]
    films_counts = [i[1] for i in sorted_films_counts]

    fig, ax1 = plt.subplots()
    bar = ax1.bar(np.arange(len(films_ind)), films_counts)
    ax1.set_xticks(np.arange(len(films_ind)))
    ax1.set_xticklabels(films_ind)
    ax1.set_ylabel("$counts$", color='b')
    ax1.set_xlabel("$movie \: j$")
    ax1.tick_params('y', colors='b')
    ax1.set_title("Frequencies and rewards of recommended films")
    for film_bar in bar:
        height = film_bar.get_height()
        ax1.text(film_bar.get_x() + film_bar.get_width()/2.0, height,'%d' % int(height), ha='center', va='bottom', color='b')

    films_rewards = [all_films_rewards[j] for j in films_ind]
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(films_rewards)), films_rewards, color='red', marker='D', linewidth=2, markersize=7)
    ax2.set_ylabel("$reward \: a_j$", color='r')
    ax2.tick_params('y', colors='r')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.grid(None)
    fig.tight_layout()
    fig.savefig("fims_freq_rewards.png", format='png', bbox_inches='tight', dpi=400)
    

def films_freq_rewards_2(films_rec, user, data, lin_ucb):
    '''bar chart of films frequency with real mean score on top.
        scores are mean values of users in the parameter'''
    films = Counter(films_rec)
    films_keys = list(films.keys())
    if len(films_keys) > 10:
        pairs = dict(sorted(films.items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        pairs = dict(sorted(films.items(), key=lambda x: x[1], reverse=True))
    pairs_keys = list(pairs.keys())
    fig, ax = plt.subplots()  
    ax.bar(np.arange(len(pairs.keys())), pairs.values())
    plt.xticks(np.arange(len(pairs.keys())), pairs_keys)
    for i, v in enumerate(pairs.values()):
        r = 0
        for u in user:
            tmp = int(data.M[lin_ucb.users[u],pairs_keys[i]]*5)
            if tmp == 0:
                tmp = data.mean_rating(i)
            r += tmp
        ax.text(i-0.2, v+0.1, "{:.1f}".format(r/len(user)), fontweight='bold')
    plt.title("film recommendation frequence")
    plt.show()
    

def all_films_rewards(all_films_rewards):
    '''Visualize rewards of all films to verify that extremely high rewards are scarce
    all_films_rewards <list> is taken from data.reward for a current user'''
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(all_films_rewards)), all_films_rewards)
    ax.set_xlabel("$movie \: j$")
    ax.set_ylabel("$r_j$")
    ax.set_title("Rewards of all films")
    fig.tight_layout()
    fig.savefig("all_films_rewards.png", format='png', bbox_inches='tight', dpi=400)

def ratings(ratings):
    '''1. ratings - T plot
       2. pie chart of frequency of ratings '''
    fig, ax = plt.subplots()
    if len(ratings) > 300:
        tmp = np.mean(np.array(ratings).reshape(-1,5),axis=1)
        ax.plot(np.arange(0,len(ratings),5), tmp)
    else:
        ax.plot(ratings)
    ax.set_xlabel("T")
    ax.set_ylabel("rating")
    ax.set_title("rating")
    fig.savefig("ratings.png", format='png', bbox_inches='tight', dpi=400)
    plt.show()
    
    ratin = Counter(ratings)
    plt.pie(ratin.values(), labels=ratin.keys(), autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('ratings')
    plt.show()
    
def rating_estimated(r_taken_mean, r_taken_ucb):
    ''' estimated scores(mean and ucb) of movies chosen by linUCB- T plot '''
    fig, ax = plt.subplots()
    ax.plot(r_taken_mean, label='mean rating')
    ax.plot(r_taken_ucb, label='ucb rating')
    ax.set_xlabel("T")
    ax.set_ylabel("rating")
    ax.set_title("estimated rating")
    plt.show()

def ratings_estimated(films_rec, ratings, flag = 'mean'):
    '''estimated scores(mean and ucb) of certain movies - T plot
        the goal is to represent the evaluation of estimation score of movies
        flag: mean or ucb'''
    films = Counter(films_rec)
    films_keys = list(films.keys())
    if len(films_keys) > 10:
        pairs = dict(sorted(films.items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        pairs = dict(sorted(films.items(), key=lambda x: x[1], reverse=True))
    pairs_keys = list(pairs.keys())
    
    for movie in pairs_keys:
        plt.plot(ratings[:,movie],label="{} rating {}".format(flag, movie))
    plt.xlabel("T")
    plt.title("estimated {} rating of films mostly recommended".format(flag))
    plt.legend()
    plt.show()