import numpy as np

# Sample lists
e_in_list = [1, 2, 3, 4, 5]
e_out_list = [5, 4, 3, 2, 1]

# Convert lists to NumPy arrays
e_in_array = np.array(e_in_list)
e_out_array = np.array(e_out_list)

# Calculate the difference
difference = e_in_array - e_out_array

# Display the result
print("Difference:", difference)
