import numpy as np
from tqdm import tqdm
import random
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.pyplot as plt

N = 200
D = 47205
TIMES = 1000
# Function to load LIBSVM data into a sparse matrix
def load_sparse_libsvm(file_path):
    features = lil_matrix((N, D))  # Initialize sparse matrix with 200 samples and 47205 features
    labels = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            split_line = line.split()
            y = int(split_line[0])  # label
            labels.append(y)
            
            for feature in split_line[1:]:
                index, value = map(float, feature.split(":"))
                features[i, int(index) - 1] = value  # Populate sparse matrix with the feature values
    
    return np.array(labels), csr_matrix(features)  # Convert to CSR format for fast operations

# PLA Algorithm with SciPy Sparse Matrices and Direct Dot Product
def pla_with_sparse_data(labels, features, max_iter=100000):
    N, d = features.shape  # N: number of examples, d: number of features
    w = np.zeros(d)  # Initialize weight vector as a dense NumPy array
    consecutive_correct = 0  # To track correct classifications over 5N examples
    target_correct = 5 * N  # Stop when we classify 5N consecutive examples correctly
    iteration = 0
    updates = 0  # To count the number of updates made
    
    while consecutive_correct < target_correct and iteration < max_iter:
        # Pick a random example
        i = random.randint(0, N - 1)
        x_n = features[i]  # This is now a sparse row in CSR format
        y_n = labels[i]
        
        # Check if the example is correctly classified using direct sparse matrix-vector multiplication
        if np.sign(x_n.dot(w)) != y_n:  # No need to convert x_n to a dense array
            # If misclassified, update the weight vector
            w += y_n * x_n.toarray().flatten()  # Update rule; convert to dense only for the update
            consecutive_correct = 0  # Reset counter on mistake
            updates += 1
        else:
            consecutive_correct += 1  # Increment counter if correctly classified
        
        iteration += 1
    
    return w, updates

def experiment(labels, features):
    updates = []

    for _ in tqdm(range(TIMES), desc="Running PLA experiments"):
        update_count = pla_with_sparse_data(labels, features)
        updates.append(update_count)
        
    # histogram of the distribution of the number of updates needed before returning wPLA
    plt.figure(figsize=(10, 6))
    plt.hist(updates, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Updates in PLA Over 1000 Runs')
    plt.xlabel('Number of Updates')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Main function to load data and run the PLA
if __name__ == '__main__':
    labels, features = load_sparse_libsvm('./rcv1_train.binary')  # Load the sparse dataset
    # pla_with_sparse_data(labels, features)
    print(features[0])
    print(features[0].toarray().shape)
    print(features[0].toarray().flatten().shape)
    # experiment(labels, features)
    # print(f"Final weight vector: {w_pla}")
    # print(f"Total updates made: {total_updates}")
