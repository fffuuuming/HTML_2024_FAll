import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm


N = 12
p = 0.15
times = 2000

def generate_data(N: int, p: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1, 1, N)
    y = np.sign(x)
    noise = np.random.choice([-1, 1], size=N, p=[p, 1-p])
    y *= noise
    return x, y

def decision_stump(x: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    thresholds = [-1] + [(x_sorted[i] + x_sorted[i+1]) / 2 for i in range(N-1) if x_sorted[i] != x_sorted[i+1]]
    
    best_ein = float('inf')
    best_s = 1
    best_theta = 0
    
    # Count positive and negative points
    total_pos = np.sum(y_sorted == 1)
    total_neg = N - total_pos

    for s in [-1, 1]:
        pos_left, neg_left = 0, 0
        pos_right, neg_right = total_pos, total_neg

        for i, theta in enumerate(thresholds):
            if i > 0:
                if y_sorted[i-1] == 1:
                    pos_left += 1
                    pos_right -= 1
                else:
                    neg_left += 1
                    neg_right -= 1

            # Calculate Ein for s = -1 and s = 1
            if s == -1:
                misclassifications = neg_left + pos_right
            else:  # s == 1
                misclassifications = pos_left + neg_right

            ein = float(misclassifications) / float(N)

            if ein < best_ein or (ein == best_ein and s * theta < best_s * best_theta):
                best_ein = ein
                best_s = s
                best_theta = theta
    
    return best_theta, best_s, best_ein

def calculate_eout(theta: float, s: int, p: float) -> float:
    return 0.5 - s * (0.5 - p) + s * (0.5 - p) * (abs(theta))

def run_experiment(num_experiments: int, N: int, p: float) -> List[Tuple[float, float]]:
    results = []
    # for _ in range(num_experiments):
    #     x, y = generate_data(N, p)
    #     theta, s, ein = decision_stump(x, y)
    #     eout = calculate_eout(theta, s, p)
    #     results.append((ein, eout))

    for _ in tqdm(range(num_experiments), desc="Running decision stump"):
        x, y = generate_data(N, p)
        theta, s, ein = decision_stump(x, y)
        eout = calculate_eout(theta, s, p)
        results.append((ein, eout))

    return results

def plot_results(results: List[Tuple[float, float]]):
    ein_values, eout_values = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.scatter(ein_values, eout_values, alpha=0.5)
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.title('Ein vs Eout for Decision Stump')
    plt.grid(True)
    plt.show()
    # plt.savefig('ein_vs_eout_scatter.png')
    # plt.close()

def main():
    results = run_experiment(times, N, p)
    plot_results(results)
    
    differences = [eout - ein for ein, eout in results]
    median_difference = np.median(differences)
    
    print(f"Median of Eout(g) - Ein(g): {median_difference:.6f}")

if __name__ == "__main__":
    main()