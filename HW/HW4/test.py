import numpy as np

def polynomial_transform(x, Q):
    poly_features = [x**q for q in range(1, Q + 1)]
    return np.hstack([np.ones((x.shape[0], 1))] + poly_features)


# # Example structure: ein_records[i][j] is the j-th recorded Ein(w_t) for the i-th experiment
# # Assuming ein_records is your list of lists with 1126 experiments, each having entries for t = 200, 400, ..., 100000
# array = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])

# # Convert to a NumPy array for easier manipulation (1126 x number of intervals)
# ein_array = np.mean(array, axis=1)

# # Display the results
# print("Average Ein(w_t) at each interval:", ein_array)



# Example input
x = np.array([[1, 2, 3], [1, 4, 5]])  # Single input with two features
Q = 3                   # Maximum polynomial degree

# Apply the polynomial transform
transformed_x = polynomial_transform(x, Q)
print("Original x:", x)
print("Transformed x:", transformed_x)