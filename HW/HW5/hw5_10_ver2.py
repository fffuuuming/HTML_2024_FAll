from liblinear.liblinearutil import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import numpy as np
import time

def load_LIBVSM(file_path):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return labels, dense_data


def experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values, iterations = 1126):
    E_out_list = []
    non_zero_count_list = []

    for _ in range(iterations):
        E_out, non_zero_count = _experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values)
        E_out_list.append(E_out)
        non_zero_count_list.append(non_zero_count)

    return E_out_list, non_zero_count_list
    

def _experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values):

    for candidate in lambda_values:

        best_lambda = lambda_values[0]
        best_acc = 0
        best_model = None
        
        # calculate C
        C = 1 / candidate
        model = train(y_train_filtered, x_train_filtered, f'-s 6 -c {C}')

        _, p_acc, _ = predict(y_train_filtered, x_train_filtered, model)

        # select best lambda with breaking tie by selecting larger lambda
        if best_acc <= p_acc[0]:
            best_acc = p_acc[0]
            best_lambda = candidate
            best_model = model

    print(f'Find the best lambda : {best_lambda}')

    # calculate E_out
    _, p_acc, _ = predict(y_test_filtered, x_test_filtered, model)
    E_out = float(1) - (p_acc[0] / float(100))

    # extract w
    w_vector = model.w[:best_model.nr_feature]

    # number of non-zero components
    non_zero_count = sum(1 for ele in w_vector if ele != 0)
    
    print(f'\nE_out : {E_out}')

    print(f'\nnon-zero count : {non_zero_count}\n')

    return E_out, non_zero_count


def plot_E_out(E_out_list, iterations):
    # Replace these with your actual lists
    # non_zero_counts = [...]
    plt.figure(figsize=(10, 6))
    plt.hist(E_out_list, bins=30, edgecolor='black')
    plt.title('Histogram of E_out')
    plt.xlabel('E_out')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"./10_eout_{iterations}.png")
    plt.close()


def plot_non_zero(non_zero_count_list, iterations):
    # Replace these with your actual lists
    # non_zero_counts = [...]
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_count_list, bins=30, edgecolor='black')
    plt.title('Histogram of non-zero components in each g')
    plt.xlabel('number of non-zero components in g')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"./10_nonzero_{iterations}.png")
    plt.close()

def main():
    start_time = time.time()

    # load dataset
    y_train, x_train = load_LIBVSM('./mnist.scale')
    y_test, x_test = load_LIBVSM('./mnist.scale.t')

    # pad x_0
    x_train = csr_matrix(np.hstack([np.ones((x_train.shape[0], 1)), x_train]))
    x_test = csr_matrix(np.hstack([np.ones((x_test.shape[0], 1)), x_test]))

    # extract class 2 & 6
    train_filter = np.where(np.isin(y_train, [2, 6]))[0]
    test_filter = np.where(np.isin(y_test, [2, 6]))[0]

    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]

    x_test_filtered = x_test[test_filter]
    y_test_filtered = y_test[test_filter]

    # convert to binary label (2 -> 1, 6 -> -1)
    y_train_filtered = np.where(y_train_filtered == 2, 1, -1)
    y_test_filtered = np.where(y_test_filtered == 2, 1, -1)

    log_lambda = [-2, -1, 0, 1, 2, 3]
    lambda_values = [10**x for x in log_lambda]

    iterations = 1126

    E_out_list, non_zero_count_list = experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values, iterations)

    plot_E_out(E_out_list, iterations)
    plot_non_zero(non_zero_count_list, iterations)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


if __name__ == "__main__":
    main()
