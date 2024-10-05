import secrets
import numpy as np
import random
import matplotlib.pyplot as plt

N = 200

def load_LIBVSM(file_path = "./rcv1_train.binary"):
    labels = []
    features = []

    with open(file_path) as f:
        for _ in range(N):
            line = f.readline()
            line = line.split()

            y = int(line[0])                        # label
            x = np.zeros(47205)                     # features

            for d in line[1:]:
                index, value = d.split(":")
                x[int(index) - 1] = float(value)    # feature index start from 1

            labels.append(y)
            features.append(x)

    return np.array(labels), np.array(features)


def pla_variant(labels, features):
    w = np.zeros(47205)                 # w_0 = 0
    # w_t = []                          # record w_t as a function of t
    consecutive_correct = 0
    update_count = 0                         
    while consecutive_correct < 5 * N:
        i = secrets.randbelow(N)        # random seed
        x_n = features[i]
        y_n = labels[i]

        product = np.dot(w, x_n)
        product = -1 if product == 0 else product   # take sign(0) as -1

        if product != y_n:
            # update w untill correct
            while product != y_n:
                update_count += 1
                w += y_n * x_n
                product = np.dot(w, x_n)
                product = -1 if product == 0 else product
        else:
            consecutive_correct += 1

    return update_count
    
def experiment(labels, features, times = 1000):
    updates = []

    for _ in range(times):
        update_count = pla_variant(labels, features)
        updates.append(update_count)

    # histogram of the distribution of the number of updates needed before returning wPLA
    plt.figure(figsize=(10, 6))
    plt.hist(updates, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Updates in PLA Over 1000 Runs')
    plt.xlabel('Number of Updates')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    labels, features = load_LIBVSM()
    experiment(labels, features)
    

