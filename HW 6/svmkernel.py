import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt

#np.random.seed(0)
Input_file = pd.read_csv('nonlinsep.txt', skipinitialspace=True, header=None, sep=',')
cols = ["x1", "x2", "label"]
Input_file.columns = cols
Input_data = Input_file[["x1", "x2"]]
Input_data = np.array(Input_data)
length = len(Input_data)
labels = Input_file[["label"]]
labels= np.array(labels)
length_labels = len(labels)


#   Create Q matrix
arr=[]
Q = np.zeros(shape=(len(Input_data), len(Input_data)))
for i in range(len(Input_data)):
    for j in range(len(Input_data)):
        lab_prod = Input_file["label"][i] * Input_file["label"][j]
        temp = np.exp(-(np.power((np.linalg.norm((Input_data[i,:] - Input_data[j,:]))), 2)) * 0.00045 )
        #arr.append(temp)
        ijth_term = lab_prod * temp
        Q[i,j] = (ijth_term)
P = matrix(Q)
q = np.ones(length) * -1.0
q = matrix(q)
b = 0.0
b = matrix(b)
G = np.diag(np.ones(length) * -1)
G = matrix(G)
h = np.zeros(length)
h = matrix(h)

labels = labels.astype(np.double)
A = matrix(labels, (1,100))

sol = solvers.qp(P, q, G, h, A, b)

sv = 0
ov = 0
above_threshold_indices = []
alphas = []
for i in range(100):
    if np.array(sol['x'])[i] > 1e-5:
        sv += 1
        above_threshold_indices.append(i)
        alphas.append(np.array(sol['x'][i]))
    else:
        ov += 1

alphas = np.transpose(np.asmatrix(alphas))

sv_datapt = []
sv_labels = []
for i in above_threshold_indices:
    sv_datapt.append(Input_data[i])
    sv_labels.append(labels[i])

#  Support Vectors
print("Support Vectors:", np.asmatrix(sv_datapt))




#   Graphing data
x1_pos = Input_file["x1"][Input_file["label"] == 1]
x2_pos = Input_file["x2"][Input_file["label"] == 1]
x1_neg = Input_file["x1"][Input_file["label"] == -1]
x2_neg = Input_file["x2"][Input_file["label"] == -1]
plt.scatter(x1_pos,x2_pos,marker='+',c='red', alpha=0.8)
plt.scatter(x1_neg, x2_neg, marker='o', c='blue', alpha=0.8)
plt.scatter(Input_data[above_threshold_indices,0],Input_data[above_threshold_indices,1], marker='*',c='black',alpha=1.0)

plt.show()


# Commented code finds the equation
"""weights2 = 0
for i in range(len(alphas)):
    temp3 = alphas[i] * labels[i]
    temp4 = np.dot(temp3, (np.asmatrix(sv_datapt[i])))
    weights2 += temp4
print("Weights 2",weights2)

#   Calculating b with weights1 and dot product
i = np.random.randint(0,100)
datapt = Input_data[i]
label = labels[i]
b = label - (np.dot(datapt, np.transpose(weights2)))"""
k = np.random.randint(0, 100)
t = 0
for i in range(len(sv_datapt)):
    temp = np.exp(-(np.power((np.linalg.norm((sv_datapt - Input_data[k,:]))), 2)) * 0.00045)
    temp2 = alphas[i] * sv_labels[i]
    t += (temp * temp2)
bias = labels[k] - t
print("Bias",bias)