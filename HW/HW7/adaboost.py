"""
todo : use Numba
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from math import sqrt
import time
import datetime

def load_LIBVSM(file_path):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return labels, dense_data


class DecisionStump:
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in [-1, 1]:

                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X[:, feature] < threshold] = -1
                    else:
                        predictions[X[:, feature] >= threshold] = -1

                    # take example weights into account
                    error = np.sum(weights[y != predictions])

                    if error < min_error:
                        min_error = error
                        self.feature_index = feature
                        self.threshold = threshold
                        self.polarity = polarity


    def predict(self, X):

        n_samples = X.shape[0]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X[:, self.feature_index] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_index] >= self.threshold] = -1
        return predictions


def adaboost(X_train, y_train, X_test, y_test, T=500):
    
    # initialize u_1 = [1/N, 1/N, ..., 1/N]
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples

    alpha_list = []
    classifiers = []
    Ein_gt_list, epsilon_t_list = [], []   # problem 10
    Ein_Gt_list, Eout_Gt_list= [], []      # problem 11
    Ut_list = [1]                          # problem 12

    for t in range(T):
        
        # train g_t
        stump = DecisionStump()
        stump.fit(X_train, y_train, weights)
        predictions = stump.predict(X_train)
 
        # compute Ein(g_t) & epsilon_t
        misclassified = (predictions != y_train).astype(int)

        Ein_gt = np.sum(misclassified) / n_samples
        epsilon_t = np.dot(weights, misclassified) / np.sum(weights)

        Ein_gt_list.append(Ein_gt)
        epsilon_t_list.append(epsilon_t)

        # compute alpha_t and update weights
        alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        weights *= np.exp(-alpha_t * y_train * predictions)
        # normalize weight, which isn't needed
        # weights /= np.sum(weights)

        # compute Ut_list
        Ut_list.append(np.sum(weights))

        alpha_list.append(alpha_t)
        classifiers.append(stump)

        # compute Ein(G_t)
        combined_predictions = np.sign(
            np.sum([alpha * clf.predict(X_train) for alpha, clf in zip(alpha_list, classifiers)], axis=0)
        )
        Ein_Gt = np.mean(combined_predictions != y_train)
        Ein_Gt_list.append(Ein_Gt)

        # compute Eout(G_t)
        combined_predictions = np.sign(
            np.sum([alpha * clf.predict(X_test) for alpha, clf in zip(alpha_list, classifiers)], axis=0)
        )
        Eout_Gt = np.mean(combined_predictions != y_test)
        Eout_Gt_list.append(Eout_Gt)

    return Ein_gt_list, epsilon_t_list, Ein_Gt_list, Eout_Gt_list, Ut_list[:-1]


def plot(list_1, list_2, list_1_name, list_2_name, num_of_prob, times):

    t = list(range(1, times + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(t, list_1, label=f'{list_1_name}', color='blue', linestyle='-', marker='o')
    plt.plot(t, list_2, label=f'{list_2_name}', color='red', linestyle='--', marker='x')

    plt.xlabel('t')
    plt.ylabel('Values')
    plt.title(f'Plot {list_1_name} and {list_2_name} as Function of t')
    plt.legend()
    plt.savefig(f'./hw7_{num_of_prob}_{times}.png')


def main():

    start_time = time.time()
    print(f'start_time : {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n')

    y_train, X_train = load_LIBVSM('./madelon')
    y_test, X_test = load_LIBVSM('./madelon.t')

    T = 500
    Ein_gt_list, epsilon_t_list, Ein_Gt_list, Eout_Gt_list, Ut_list = adaboost(X_train, y_train, X_test, y_test, T)

    # problem 10, 11, 12
    plot(Ein_gt_list, epsilon_t_list, 'Ein_gt_list', 'epsilon_t_list', 10, T)
    plot(Ein_Gt_list, Eout_Gt_list, 'Ein_Gt_list', 'Eout_Gt_list', 11, T)
    plot(Ein_Gt_list, Ut_list, 'Ein_Gt_list', 'Ut_list', 12, T)


    end_time = time.time()
    execution_time = end_time - start_time

    print(f'start_time : {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n')
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()

