import numpy as np

# Path to your .npy file
file_path = 'OUTPUT_DIR/predict_01_with_roberta_layer_6_len_4.npy'

# Load the data
data = np.load(file_path, allow_pickle=True)

# Print the contents or shape
print("Data shape:", data.shape)
print("Data contents:", data)