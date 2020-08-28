import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


Input_file = pd.read_csv("pca-data.txt", skipinitialspace=True, header=None, sep="[\t]", engine= 'python')
cols = ["x1", "x2", "x3"]
Input_file.columns = cols
data_vectors = []
for i in range(len(Input_file)):
    data_vectors.append(np.array([Input_file.iloc[i,0] , Input_file.iloc[i,1] , Input_file.iloc[i,2]]))
data_vectors = np.array(data_vectors)
def PCA(k = 2):
    fig2 = plt.figure()
    covariance_matrix = average_and_covariance(data_vectors)
    evals, evecs = np.linalg.eig(covariance_matrix)
    k_evals = np.argsort(evals)[::-1]
    k_evals = k_evals[0:k]
    k_evecs = evecs[:,k_evals[0:k]]
    reduced_dimensional_data_vectors = []
    for i in range(data_vectors.shape[0]):
        z = np.transpose(k_evecs) * np.transpose(np.asmatrix(data_vectors[i]))
        reduced_dimensional_data_vectors.append(z)

    return k_evecs, reduced_dimensional_data_vectors
def average_and_covariance(data):
    avg = 0
    for i in range(data.shape[0]):
        avg += data[i]
    avg = avg / data.shape[0]
    data = np.asmatrix(data)
    cov = 0
    normalized_data = data - avg
    cov = np.dot(np.transpose(normalized_data), normalized_data)
    cov = cov / data.shape[0]

    return cov
print(PCA()[0])
reduced_data = PCA()[1]
reduced_data = np.array(reduced_data)
for i in range(len(Input_file)):
    x = reduced_data[i][0]
    y = reduced_data[i][1]
    plt.scatter(x, y)
plt.savefig("Reduced Dimensional Data")
plt.close()

#Plotting The original 3D data

fig = plt.figure()
ax = fig.gca(projection = '3d')
for i in range(data_vectors.shape[0]):
    x_coordinate = data_vectors[i,0]
    y_coordinate = data_vectors[i,1]
    z_coordinate = data_vectors[i,2]
    ax.scatter(x_coordinate, y_coordinate, z_coordinate)
fig.savefig("Original Data")