import numpy as np
import pandas as pd
Input_file = pd.read_csv('linear-regression.txt', header=None, skipinitialspace=True, sep=',')
Input_file = Input_file.iloc[:,0:3]
cols = ['x1', 'x2', 'y']
Input_file.columns = cols
output_vals = np.transpose(np.asmatrix(Input_file["y"]))
data_matrix = np.zeros((len(Input_file), 3))
for i in range(len(data_matrix)):
    data_matrix[i] = (np.append([1.0] , Input_file.iloc[i, 0:2]))
data_matrix = np.transpose(data_matrix)

#Computing matrices
data_data_transpose = np.dot(data_matrix, np.transpose(data_matrix))
data_output_vals = np.dot(data_matrix, output_vals)
weights = np.dot(np.linalg.inv(data_data_transpose), data_output_vals)
print(weights)