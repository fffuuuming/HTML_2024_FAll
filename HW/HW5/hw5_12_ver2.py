from liblinear.liblinearutil import *
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import numpy as np
import time

def load_LIBVSM(file_path):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return labels, dense_data


def experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values, iterations = 1126):
    E_out_list = []

    for _ in range(iterations):
        E_out_list.append(_experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values))

    return E_out_list
    

def _experiment(y_train, x_train, y_test, x_test, lambda_values):
    
    # find best lambda
    for candidate in lambda_values:

        best_lambda = lambda_values[0]
        best_acc = 0
        
        # calculate C
        C = 1 / candidate
        p_acc = train(y_train, x_train, f'-s 6 -c {C} -v 3')

        # select best lambda with breaking tie by selecting larger lambda
        if best_acc <= p_acc:
            best_acc = p_acc
            best_lambda = candidate

    # print(f'Find the best lambda : {best_lambda}')

    # re-run on the whole training set with the best lambda
    model = train(y_train, x_train, f'-s 6 -c {1 / best_lambda}')

    _, p_acc, _ = predict(y_test, x_test, model)

    E_out = float(1) - (p_acc[0] / float(100))
    
    # print(f'E_out : {E_out}')

    return E_out


def plot_E_out(E_out_list, iterations):

    plt.figure(figsize=(10, 6))
    plt.hist(E_out_list, bins=30, edgecolor='black')
    plt.title('Histogram of E_out')
    plt.xlabel('E_out')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"./12_{iterations}.png")
    plt.close()


def main():
    start_time = time.time()

    # load dataset
    y_train, x_train = load_LIBVSM('./mnist.scale')
    y_test, x_test = load_LIBVSM('./mnist.scale.t')

    # pad x_0
    x_train = csr_matrix(np.hstack([np.ones((x_train.shape[0], 1)), x_train]))
    x_test = csr_matrix(np.hstack([np.ones((x_test.shape[0], 1)), x_test]))

    log_lambda = [-2, -1, 0, 1, 2, 3]
    lambda_values = [10**x for x in log_lambda]

    iterations = 1126

    E_out_list = experiment(y_train, x_train, y_test, x_test, lambda_values, iterations)

    plot_E_out(E_out_list, iterations)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()
