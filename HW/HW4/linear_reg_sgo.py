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

def sgd_linear_regression(train_data, train_target, validation_data, validation_target, eta=0.01, iterations=100000, interval=200):

    # N = 64, d = 13
    size, d = train_data.shape

    w_t = np.zeros(d)
    ein_sgd_records = []
    eout_sgd_records = []
    
    for t in range(1, iterations + 1):

        i = np.random.choice(np.arange(size))
        xi = train_data[i]
        yi = train_target[i]
        
        gradient = 2 * (yi - xi @ w_t) * xi
        w_t += eta * gradient
        
        if t % interval == 0:
            ein_t = np.mean((train_data @ w_t - train_target) ** 2)
            eout_t = np.mean((validation_data @ w_t - validation_target) ** 2)
            ein_sgd_records.append(ein_t)
            eout_sgd_records.append(eout_t)
    
    return ein_sgd_records, eout_sgd_records

def polynomial_transform(x, Q):
    poly_features = [x**q for q in range(1, Q + 1)]
    return np.hstack([np.ones((x.shape[0], 1))] + poly_features)


def sgd_exepriments(X, y, N, times):

    Ein_wlin_list = []
    Eout_wlin_list = []

    Ein_sgd_records_list = []
    Eout_sgd_records_list = [] 

    length = len(X)

    for _ in tqdm(range(times), desc="Running Linear Regression"):

        # training data
        train_indices = np.random.choice(np.arange(length), size=N, replace=False)
        X_train, y_train = X[train_indices], y[train_indices]

        # testing data 
        test_indices = list(set(range(len(X))) - set(train_indices))
        X_test, y_test = X[test_indices], y[test_indices]

        # w_lin
        w_lin = linear_regression(X_train, y_train)
        
        # Ein(w_lin), Eout(w_out)
        Ein = mean_squared_error(X_train, y_train, w_lin)
        Eout = mean_squared_error(X_test, y_test, w_lin)

        Ein_wlin_list.append(Ein)
        Eout_wlin_list.append(Eout)

        # SGD
        Ein_sgd_records, Eout_sgd_records = sgd_linear_regression(X_train, y_train, X_test, y_test)

        Ein_sgd_records_list.append(Ein_sgd_records)
        Eout_sgd_records_list.append(Eout_sgd_records)


    Ein_wlin_mean = np.mean(Ein_wlin_list)
    Eout_wlin_mean = np.mean(Eout_wlin_list)

    Ein_sgd_mean_list = np.mean(Ein_sgd_records_list, axis=0)
    Eout_sgd_mean_list = np.mean(Eout_sgd_records_list, axis=0)

    return Ein_wlin_mean, Eout_wlin_mean, Ein_sgd_mean_list, Eout_sgd_mean_list

def exepriments(X, y, N, times):

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

def plot(Ein_wlin_mean, Eout_wlin_mean, Ein_sgd_mean_list, Eout_sgd_mean_list):
    # t = 200, 400, 600, ..., 100000
    t_values = np.arange(200, 200 * (len(Ein_sgd_mean_list) + 1), 200)

    plt.figure(figsize=(10, 6))

    plt.plot(t_values, Ein_sgd_mean_list, label="Average Ein(w_t)", color="blue", marker="o")
    plt.plot(t_values, Eout_sgd_mean_list, label="Average Eout(w_t)", color="red", marker="x")

    plt.axhline(y=Ein_wlin_mean, color="blue", linestyle="--", label="Average Ein(w_lin)")
    plt.axhline(y=Eout_wlin_mean, color="red", linestyle="--", label="Average Eout(w_lin)")

    plt.xlabel("Iterations (t)")
    plt.ylabel("Error")
    plt.title("Average Ein(w_t) and Eout(w_t) over Iterations")
    plt.legend()

    # # Show plot
    # plt.grid()
    # plt.show()

    # Save plot
    plt.grid()
    plt.savefig("./10_sgo.png")
    plt.close()

def plot_Ein_gain(Ein_wlin_list, Ein_wpoly_list):
    
    gain = np.array(Ein_wlin_list) - np.array(Ein_wpoly_list)
    mean_gain = np.mean(gain)

    plt.figure(figsize=(8, 6))
    plt.hist(gain, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Ein(wlin) - Ein(wpoly)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Ein(wlin) - Ein(wpoly)")

    # # show plot
    # plt.grid(True)
    # plt.show()

    # save plot
    plt.grid()
    plt.savefig("./11_poly_transfrom_Ein.png")
    plt.close()

    print(f"Average gain: {mean_gain}")

def plot_Eout_change(Eout_wlin_list, Eout_wpoly_list):

    change = np.array(Eout_wlin_list) - np.array(Eout_wpoly_list)
    mean_change = np.mean(change)

    plt.figure(figsize=(8, 6))
    plt.hist(change, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Eout(wlin) - Eout(wpoly)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Eout(wlin) - Eout(wpoly)")

    # # show plot
    # plt.grid(True)
    # plt.show()

    # save plot
    plt.grid()
    plt.savefig("./12_poly_transfrom_Eout.png")
    plt.close()

    print(f"Average gain: {mean_change}")


def main():

    dense_data, labels = load_LIBVSM()

    # add x_0
    # number of data, feature : (8192, 13)
    X = np.hstack([np.ones((dense_data.shape[0], 1)), dense_data])

    # set the parameter
    N, times = 64, 50

    # problem 10
    # Ein_wlin_mean, Eout_wlin_mean, Ein_sgd_mean_list, Eout_sgd_mean_list = sgd_exepriments(X, labels, N, times)
    # plot(Ein_wlin_mean, Eout_wlin_mean, Ein_sgd_mean_list, Eout_sgd_mean_list)

    times = 1126

    # Q-order polynomial transform
    X_poly = polynomial_transform(dense_data, 3)

    Ein_wlin_list, Eout_wlin_list = exepriments(X, labels, N, times)
    Ein_wpoly_list, Eout_wpoly_list = exepriments(X_poly, labels, N, times)


    # problem 11
    plot_Ein_gain(Ein_wlin_list, Ein_wpoly_list)


    # problem 12
    plot_Eout_change(Eout_wlin_list, Eout_wpoly_list)


if __name__ == "__main__":
    main()
