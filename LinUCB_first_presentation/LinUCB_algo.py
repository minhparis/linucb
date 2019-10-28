from sklearn.preprocessing import normalize
import math
import numpy as np

class LinUCB:
    def __init__(self, movielens_data, context = 'user', alpha = 1):
        self.data = movielens_data
        # delta = 0.2
        # self.alpha = 1 + math.sqrt(math.log(2/delta)/2)
        self.alpha = alpha
        self.context = context
        
        self.n_movies = movielens_data.n_movies
        self.users = movielens_data.active_users()
        self.n_users = self.users.shape[0]
        
        if context == 'user':
            self.X = np.concatenate((movielens_data.U[self.users], np.ones((self.n_users,1))), axis = 1)
        else:
            self.X = np.concatenate((movielens_data.Vt.transpose(), np.ones((self.n_movies,1))), axis = 1)
        self.X = normalize(self.X, axis=0)

        self.d = self.X.shape[1]
        
        self.A = np.repeat(np.identity(self.d)[np.newaxis, :, :], self.n_movies, axis=0)
        self.b = np.repeat(np.zeros(self.d)[np.newaxis, :], self.n_movies, axis=0)
        
        self.recom_record = np.zeros((self.n_users, self.n_movies))
        self.theta = np.zeros((self.n_movies, self.d))

        self.first_add = np.zeros((self.n_movies))

    def choose_arm(self, user):
        p_t = np.zeros((self.n_movies))
        self.theta = np.zeros((self.n_movies, self.d))
        
        if self.context == 'user':
            x = self.X[user,:].reshape(-1,1)
        for a in range(self.n_movies):
            if self.context != 'user':
                x = self.X[a,:].reshape(-1,1)
            if self.recom_record[user, a] >= 1:
                p_t[a] = -1e2
                continue
            r = self.data.reward(self.users[user], a)
            if r == 0:
                p_t[a] = -1e2
                continue
            A_a_inv = np.linalg.inv(self.A[a])
            self.theta[a,:] = np.dot(A_a_inv, self.b[a].transpose()).transpose()
            p_t[a] = np.dot(self.theta[a,:], x) + self.alpha * np.sqrt(np.dot(np.dot(x.transpose(), A_a_inv), x))
        #choose arm
        best_arms = np.flatnonzero(p_t == p_t.max())
        # print('pta: \n{}'.format(p_t))
        # print('max: {}, \nfilm: \n{}'.format(p_t.max(), best_arms))
        a = np.random.choice(best_arms) # choose with ties broken
        return a
    
    def fit(self, l_users):
        regrets = []
        ratings = []
        self.recom_record = np.zeros((self.n_users, self.n_movies))
        film_rec_count = np.zeros((self.n_movies))
        for i in range(50):
            rating = 0
            regret = 0
            n = 0
            
            for user in l_users:

                a = self.choose_arm(user)
                film_rec_count[a] += 1
                if self.recom_record[user, a] < 1:
                    self.recom_record[user, a] += 1
                    
                    if self.context == 'user':
                        x = self.X[user,:].reshape(-1,1)
                    else:
                        x = self.X[a,:].reshape(-1,1)
                    pred_rating = np.dot(self.theta[a,:].transpose(), x)
                    r = self.data.reward(self.users[user], a)
                    if r == 0:
                        print('this film has not a score')
                        continue
                    self.A[a] += np.dot(x, x.transpose())
                    self.b[a] += ((r - pred_rating) * x).flatten()
                    # self.b[a] += r * x.transpose()
                    if self.first_add[a] == 0:
                        self.first_add[a] = i+1
                    regret += (r - pred_rating) ** 2
                    rating += r
                    n += 1
                    
            if n == 0:
                continue
            regrets.append(math.sqrt(regret / n))
            ratings.append(rating / n)
        return regrets, ratings, film_rec_count


    def evaluator(self, l_users):
        regrets = []
        ratings = []
        self.recom_record = np.zeros((self.n_users, self.n_movies))
        film_rec_count = np.zeros((self.n_movies))

        for i in range(50):
            rating = 0
            regret = 0
            n = 0
            
            for user in l_users:
                a = self.choose_arm(user)
                film_rec_count[a] += 1
                if self.recom_record[user, a] < 1:
                    self.recom_record[user, a] += 1
                    
                    if self.context == 'user':
                        x = self.X[user,:].reshape(-1,1)
                    else:
                        x = self.X[a,:].reshape(-1,1)
                    pred_rating = np.dot(self.theta[a,:].transpose(), x)
                    r = self.data.reward(self.users[user], a)
                    if r == 0:
                        continue
                    
                    if self.first_add[a] == 0:
                        self.first_add[a] = i+1
                    regret += (r - pred_rating) ** 2
                    rating += r
                    n += 1

            if n == 0:
                continue

            regrets.append(math.sqrt(regret / n))
            ratings.append(rating / n)
        return regrets, ratings, film_rec_count