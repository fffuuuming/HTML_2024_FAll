import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

N = 12
p = 0.15
times = 2000

def generate_data():
    x = np.random.uniform(-1, 1, N)
    y = np.sign(x)
    noise = np.random.choice([-1, 1], size=N, p = [p, 1 - p])
    y *= noise
    return x, y

def decision_stump(x, y):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    thresholds = [-1]
    for i in range(N-1):
        if x_sorted[i] != x_sorted[i+1]:
            thresholds.append((x_sorted[i] + x_sorted[i+1]) / 2)

    best_ein = float('inf')
    best_s = 1
    best_theta = 1

    total_pos = np.sum(y_sorted == 1)
    tota_neg = N - total_pos

    for s in [-1, 1]:
        pos_left, neg_left = 0, 0
        pos_right, neg_right = total_pos, tota_neg

        length = len(thresholds)
        
        for i in range(1, length):
            if y_sorted[i-1] == 1:
                pos_left += 1
                pos_right -= 1
            else:
                neg_left += 1
                neg_right -= 1
            
            if s == -1:
                e_in = float(neg_left + pos_right) / float(N)
            else:
                e_in = float(pos_left + neg_right) / float(N)
            
            theta = thresholds[i]

            if e_in < best_ein or (e_in == best_ein and s * theta < best_s * best_theta):
                best_ein = e_in
                best_s = s
                best_theta = (thresholds[i] + thresholds[i-1]) / 2

    return best_theta, best_s, best_ein


def decision_stump_random(x, y):
    # randomly generate theta and s
    theta = np.random.uniform(-1, 1, N)
    s = np.random.choice([-1, 1], size=None)

    missclassification = 0
    for i in range(N):
        if(y[i] != np.sign(x[i] - theta) * s):
            missclassification += 1

    e_in = missclassification / N

    return theta, s, e_in

def calculate_eout(theta, s, p):
    return 0.5 - s * (0.5 - p) + s * (0.5 - p) * (abs(theta))

def main():
    decision_stump("test", 0)


if __name__ == "__main__":
    main()