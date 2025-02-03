from libsvm.svmutil import *
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time

def load_LIBVSM(file_path):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return labels, dense_data


def experiments(y_train_filtered, x_train_filtered, c, gamma_candidates, times = 128):
    frequency = {
        0.01: 0,
        0.1: 0,
        1: 0,
        10: 0,
        100: 0
    }

    for _ in range(times):
        # frequency[best_gamma] += 1
        frequency[_experiment(y_train_filtered, x_train_filtered, c, gamma_candidates)] += 1
    
    return frequency
        
def _experiment(y_train_filtered, x_train_filtered, c, gamma_candidates):

    # randonly samples 200 examples for validation
    x_sub_train, x_val, y_sub_train, y_val = train_test_split(x_train_filtered, y_train_filtered, test_size=200)

    best_gamma = 0
    best_acc = 0

    # find the best gamma
    for r in gamma_candidates:

        model = svm_train(y_sub_train, x_sub_train, f'-s 0 -c {c} -t 2 -g {r}')
        _, p_acc, _ = svm_predict(y_val, x_val, model)

        if best_acc <= p_acc[0]:
            best_acc = p_acc[0]
            best_gamma = r
            
    return best_gamma


def plot_bar_chart(frequency):

    keys = list(frequency.keys())
    values = list(frequency.values())

    plt.bar(range(len(keys)), values, width=0.5, color='skyblue')

    plt.xlabel('γ values')
    plt.ylabel('Selection Frequency')
    plt.title('Frequency of γ Selections')
    plt.xticks(range(len(keys)), keys)

    plt.savefig('./hw6_12.png')


def main():

    start_time = time.time()

    y_train, x_train = load_LIBVSM('./mnist.scale')

    train_filter = np.where(np.isin(y_train, [3, 7]))[0]

    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]

    # convert to binary label (3 -> 1, 7 -> -1)
    y_train_filtered = np.where(y_train_filtered == 3, 1, -1)
    
    C = 1
    gamma_candidates = [100, 10, 1, 0.1, 0.01]
    times = 128

    frequency = experiments(y_train_filtered, x_train_filtered, C, gamma_candidates, times)

    plot_bar_chart(frequency)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()

