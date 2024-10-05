import secrets
from tqdm import tqdm
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt

N = 200
D = 47205
TIMES = 1000

features = [[2,2], [5,4]]
labels = [1, -1]

def pla(labels, features):
    w = csr_matrix((1,D))               # w_0 = 0
    # print(w.shape)
    # w_t = []                          # record w_t as a function of t 
    consecutive_correct = 0
    update_count = 0

    while consecutive_correct < 5 * N:
        i = secrets.randbelow(N)        # random seed
        x_n = features[i]
        y_n = labels[i]

        # print(x_n.shape)
        if np.sign(x_n.dot(w.T)) != y_n:
            consecutive_correct = 0
            update_count += 1
            w += y_n * x_n
        else:
            consecutive_correct += 1

    return update_count
    
def experiment(labels, features):
    updates = []

    for _ in tqdm(range(TIMES), desc="Running PLA experiments"):
        update_count = pla(labels, features)
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
    print(features[0])
    # labels, features = load_LIBVSM()
    # # print(labels)
    # # print(features)
    # experiment(labels, features)
    

0xb794f5ea0ba39494ce839613fffba74279579268