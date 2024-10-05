def read_libsvm_file(file_path):
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            if line_num == 5: break
            # Split the line into label and features
            split_line = line.split()
            label = split_line[0]  # First element is the label
            features = split_line[1:]  # The rest are features

            # Prepare the features for printing
            feature_dict = {}
            for feature in features:
                index, value = feature.split(":")
                feature_dict[int(index)] = float(value)

            # Print the label and feature dictionary
            print(f"Line {line_num}:")
            print(f"  Label: {label}")
            print(f"  Features: {feature_dict}")
            print()  # Blank line between entries

if __name__ == '__main__':
    file_path = './rcv1_train.binary'  # Replace with the actual path
    read_libsvm_file(file_path)
