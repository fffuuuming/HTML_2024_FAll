from liblinear.liblinearutil import *
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

def load_LIBVSM(file_path):
    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    return labels, dense_data

def experiment(y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values, iterations=1126, num_cpus=5):
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [
            executor.submit(_experiment, y_train_filtered, x_train_filtered, y_test_filtered, x_test_filtered, lambda_values, seed=np.random.randint(0, 10000))
            for _ in range(iterations)
        ]
        E_out_list = [future.result() for future in futures]
    return E_out_list

def _experiment(y_train, x_train, y_test, x_test, lambda_values, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Find best lambda
    for candidate in lambda_values:

        best_lambda = lambda_values[0]
        best_acc = 0

        # Calculate C
        C = 1 / candidate
        p_acc = train(y_train, x_train, f'-s 6 -c {C} -v 3')

        # Select best lambda with breaking tie by selecting larger lambda
        if best_acc <= p_acc:
            best_acc = p_acc
            best_lambda = candidate

    # Re-run on the whole training set with the best lambda
    model = train(y_train, x_train, f'-s 6 -c {1 / best_lambda}')
    _, p_acc, _ = predict(y_test, x_test, model)
    E_out = float(1) - (p_acc[0] / float(100))
    return E_out

def plot_E_out(E_out_list, iterations):
    plt.figure(figsize=(10, 6))
    plt.hist(E_out_list, bins=30, edgecolor='black')
    plt.title('Histogram of E_out')
    plt.xlabel('E_out')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(f"./12_{iterations}_parallel.png")
    plt.close()

def main():
    start_time = time.time()

    # Load dataset
    y_train, x_train = load_LIBVSM('./mnist.scale')
    y_test, x_test = load_LIBVSM('./mnist.scale.t')

    # Pad x_0
    x_train = csr_matrix(np.hstack([np.ones((x_train.shape[0], 1)), x_train]))
    x_test = csr_matrix(np.hstack([np.ones((x_test.shape[0], 1)), x_test]))

    log_lambda = [-2, -1, 0, 1, 2, 3]
    lambda_values = [10**x for x in log_lambda]

    iterations = 1126

    # Run the experiment with parallel processing
    E_out_list = experiment(y_train, x_train, y_test, x_test, lambda_values, iterations)

    # Plot E_out histogram
    plot_E_out(E_out_list, iterations)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()
