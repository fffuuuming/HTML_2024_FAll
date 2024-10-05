"""
PLA with Histogram

"""

import secrets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_LIBVSM(file_path = './rcv1_train.binary', lines = 200):
    features = []
    labels = []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= lines: break

            data = line.split()
            y = int(data[0])
            x = np.zeros(47205)

            for d in data[1:]:
                index, value = d.split(":")
                x[int(index) - 1] = float(value)    # feature index start from 1

            labels.append(y)
            features.append(x)

    return np.array(features), np.array(labels)

# PLA
def pla(features, labels, T_min):
    N, d = features.shape   # N points ith d fetures
    w = np.zeros(d)         # w_0 = 0
    w_t = []                # record w_t as a function of t 

    consecutive_correct = 0
    target_correct = 5 * N
    updates = 0
    
    while consecutive_correct < target_correct:

        # random select x_n & y_n
        i = secrets.randbelow(N)
        x_n = features[i]
        y_n = labels[i]

        # if w is incorrect
        if np.sign(np.dot(w, x_n)) != y_n:
            updates += 1                        # increase number of updates
            consecutive_correct = 0             # reset consecutive_correct
            w += y_n * x_n                      # updates w_t

            # record w_t if updates <= T_min
            if updates <= T_min:
                w_t.append(np.linalg.norm(w))

        else:
            consecutive_correct += 1

    T_min = min(T_min, updates)                 # updates T_min
    
    return(updates, T_min, w_t)



def experiment(data, labels, num_experiments=1000):

    updates_list = []           # number of updates for each experiment
    T_min = float('inf')   # smallest number of updates in the previous problem
    func_w_t = []          # all fucntions of w_t

    for _ in tqdm(range(num_experiments), desc="Running PLA experiments"):
        updates, T_min, w_t = pla(data, labels, T_min)
        updates_list.append(updates)

    # for _ in range(num_experiments):
    #     res = pla(data, labels, T_min)
    #     updates.append(res[0])
    #     T_min = res[1]
    #     func_w_t.append(res[2])
    
    # histogram of the distribution of the number of updates needed before returning wPLA
    plt.figure(figsize=(10, 6))
    plt.hist(updates_list, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Updates in PLA Over 1000 Runs')
    plt.xlabel('Number of Updates')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # # histogram of the norm of each experiment
    # plt.figure(figsize=(10, 6)) 
    # for w_t in func_w_t:
    #     plt.plot(w_t, alpha=0.2, color='blue')  # Superpose the 1000 functions
    
    # plt.title(f'Norm of Weight Vector (∥w_t∥) as a Function of t')
    # plt.xlabel('t (Number of Updates)')
    # plt.ylabel('∥w_t∥ (Norm of Weight Vector)')
    # plt.grid(True)    
    # plt.show()

if __name__ == '__main__':
    data, labels = load_LIBVSM('./rcv1_train.binary')
    experiment(data, labels)
