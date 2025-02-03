from libsvm.svmutil import *
from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np
import time

def load_LIBVSM(file_path):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return labels, dense_data


def _experiment(y_train_filtered, x_train_filtered, C_candidates, Q_candidates):
    min_num_sv = {'C': 0, 'Q \\ C': 0, 'Number of Support Vectors': float('inf')}
    num_sv_list = []

    for _, q in enumerate(Q_candidates):
        for _, c in enumerate(C_candidates):

            # soft SVM, kernel = (γu^Tv + coef0)^degree -> (1 + x_n^Tx_m)^Q
            model = svm_train(y_train_filtered, x_train_filtered, f'-s 0 -c {c} -t 1 -d {q} -g 1 -r 1')
            num_sv = {'C': c, 'Q \\ C': q, 'Number of Support Vectors': model.get_nr_sv()}
            
            if min_num_sv['Number of Support Vectors'] > num_sv['Number of Support Vectors']:
                min_num_sv = num_sv

            num_sv_list.append(num_sv)

        # return num_sv_list
    
    return num_sv_list, min_num_sv


def construct_table(num_sv_list):
    df = pd.DataFrame(num_sv_list)
    pivot_df = df.pivot(index="Q \\ C", columns="C", values="Number of Support Vectors")
    pivot_df.to_csv('number_of_sv.csv')


def main():
    start_time = time.time()

    y_train, x_train = load_LIBVSM('./mnist.scale')

    train_filter = np.where(np.isin(y_train, [3, 7]))[0]

    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]

    # convert to binary label (3 -> 1, 7 -> -1)
    y_train_filtered = np.where(y_train_filtered == 3, 1, -1)

    # print(y_train_filtered)
    # print(x_train_filtered)
    
    Q_candidates = [2, 3, 4]
    C_candidates = [0.1, 1, 10]

    num_sv_list, min_num_sv = _experiment(y_train_filtered, x_train_filtered, C_candidates, Q_candidates)

    print(f'\nAt C = {min_num_sv['C']}, Q = {min_num_sv['Q \\ C']}, number of support vectors has the smallest value : {min_num_sv['Number of Support Vectors']}\n')

    construct_table(num_sv_list)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()

