from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time

def load_LIBVSM(file_path):

    data, labels = load_svmlight_file(file_path)
    dense_data = data.toarray()
    
    return labels, dense_data


def compute_gaussian_kernel_margin(sv, coef, gamma):
    """
    w^Tw = alpha^TQalpha
    To calculate kernel matrix Q : Q_{i,j} = K(x_i, x_j) 
                                           = exp(-gamma||x_i - x_j||^2)
    ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2x^Tx'
    """
    
    # Compute the squared norms of each sample
    sq_norms = np.sum(sv ** 2, axis=1).reshape(-1, 1)

    # Compute pairwise squared distances and then apply rbf
    kernel_matrix = np.exp((-gamma) * (sq_norms + sq_norms.T - 2 * np.dot(sv, sv.T)))

    # w^Tw = alpha^TQalpha
    norm_w_squared = np.dot(coef, np.dot(kernel_matrix, coef))
    margin = 1 / np.sqrt(norm_w_squared)

    return margin


def _experiment(y_train_filtered, x_train_filtered, C_candidates, gamma_candidates):
    max_margin = {'C': 0, 'gamma \\ C': 0, 'margin': 0}
    margin_list = []

    for _, r in enumerate(gamma_candidates):
        for _, c in enumerate(C_candidates):
            
            print(f'\n gamma = {r}, c = {c}:\n')

            model = SVC(kernel='rbf', C=c, gamma=r)
            model.fit(x_train_filtered, y_train_filtered)
            
            print(f'model computed\n')

            sv = model.support_vectors_
            coef = model.dual_coef_[0]

            margin = {'C': c, 'gamma \\ C': r, 'margin': compute_gaussian_kernel_margin(sv, coef, r)}
            
            print('margin computed\n')
            
            if max_margin['margin'] < margin['margin']:
                max_margin = margin

            margin_list.append(margin)
            
    return margin_list, max_margin


def construct_table(margin_list):
    df = pd.DataFrame(margin_list)
    pivot_df = df.pivot(index="gamma \\ C", columns="C", values="margin")
    pivot_df.to_csv('margin.csv')

    
def main():

    start_time = time.time()

    y_train, x_train = load_LIBVSM('./mnist.scale')

    train_filter = np.where(np.isin(y_train, [3, 7]))[0]

    x_train_filtered = x_train[train_filter]
    y_train_filtered = y_train[train_filter]

    # convert to binary label (3 -> 1, 7 -> -1)
    y_train_filtered = np.where(y_train_filtered == 3, 1, -1)
    
    gamma_candidates = [0.1, 1, 10]
    C_candidates = [0.1, 1, 10]

    margin_list, max_margin = _experiment(y_train_filtered, x_train_filtered, C_candidates, gamma_candidates)

    print(f'\nAt C = {max_margin['C']}, gamma = {max_margin['gamma \\ C']}, largest margin : {max_margin['margin']}\n')

    construct_table(margin_list)
    
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

if __name__ == "__main__":
    main()

