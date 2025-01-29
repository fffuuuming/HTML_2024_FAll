from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def load_LIBVSM(file_path = './cpusmall_scale'):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return dense_data, labels


def linear_regression(X, y):

    X_pseudo_inv = np.linalg.pinv(X)
    w_lin = X_pseudo_inv @ y

    return w_lin


def mean_squared_error(X, y, w):

    predictions = X @ w
    errors = predictions - y

    return np.mean(errors ** 2)


def exepriments(X, y, times):
    
    Ein_avg_list, Eout_avg_list = [], []

    for i in range(25, 2001, 25):
        Ein_list, Eout_list = _exepriments(X, y, i, times)

        Ein_avg = sum(Ein_list) / len(Ein_list)
        Eout_avg = sum(Eout_list) / len(Eout_list)

        Ein_avg_list.append(Ein_avg)
        Eout_avg_list.append(Eout_avg)

    return Ein_avg_list, Eout_avg_list


def _exepriments(X, y, N, times):

    Ein_list = []
    Eout_list = []

    length = len(X)

    for _ in tqdm(range(times), desc="Running Linear Regression"):

        # randomly selected examples
        train_indices = np.random.choice(np.arange(length), size=N, replace=False)
        X_train, y_train = X[train_indices], y[train_indices]

        # construct the remaining examples
        test_indices = list(set(range(len(X))) - set(train_indices))
        X_test, y_test = X[test_indices], y[test_indices]
    
        w_lin = linear_regression(X_train, y_train)

        Ein = mean_squared_error(X_train, y_train, w_lin)
        Eout = mean_squared_error(X_test, y_test, w_lin)

        Ein_list.append(Ein)
        Eout_list.append(Eout)

    return Ein_list, Eout_list

def plot_scatter(Ein_list, Eout_list):

    plt.figure(figsize=(10, 6))
    plt.scatter(Ein_list, Eout_list, alpha=0.5)
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.title('Ein vs Eout for Linear Regression')
    plt.grid(True)
    plt.show()

def plot_learning_curve(Ein_avg_list, Eout_avg_list):

    N_values = list(range(25, 2001, 25))

    plt.figure(figsize=(10, 6))
    plt.plot(N_values, Ein_avg_list, label="Average Ein (N)", color='blue', marker='o')
    plt.plot(N_values, Eout_avg_list, label="Average Eout (N)", color='red', marker='x')
    plt.xlabel("N (Number of Training Examples)")
    plt.ylabel("Error")
    plt.title("Learning Curves: Average Ein(N) and Eout(N) vs N")
    plt.legend()
    plt.show()


def main():

    dense_data, labels = load_LIBVSM()

    # add x_0
    X = np.hstack([np.ones((dense_data.shape[0], 1)), dense_data])

    # set the parameter
    N, times = 32, 1126

    # problem 10
    Ein_list, Eout_list = _exepriments(X, labels, N, times)
    plot_scatter(Ein_list, Eout_list)

    # problem 11
    times = 16
    Ein_list, Eout_list = exepriments(X, labels, times)
    plot_learning_curve(Ein_list, Eout_list)
    
    # problem 12
    # slice the matrix
    X_reduced = X[:, :3]

    Ein_list, Eout_list = exepriments(X_reduced, labels, times)
    plot_learning_curve(Ein_list, Eout_list)


if __name__ == "__main__":
    main()
