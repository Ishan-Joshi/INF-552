import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
np.random.seed(0)
Input_file = pd.read_csv('linsep.txt', skipinitialspace=True, header=None, sep=',')
cols = ["x1", "x2", "label"]
Input_file.columns = cols
Input_data = Input_file[["x1", "x2"]]
Input_data = np.array(Input_data)
length = len(Input_data)
labels = Input_file[["label"]]
labels= np.array(labels)
length_labels = len(labels)




#   Create Q matrix
Q = np.zeros(shape=(len(Input_data), len(Input_data)))
for n in range(len(Input_file)):
    for m in range(len(Input_file)):
        lab_prod = Input_file["label"][n] * Input_file["label"][m]
        datapoint_prod = np.dot(np.transpose(Input_data[n]), Input_data[m])
        mnth_term = lab_prod * datapoint_prod
        Q[n,m] = (mnth_term)
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

print("Support Vectors:", sv_datapt)

weights2 = 0
for i in range(len(alphas)):
    temp3 = alphas[i] * sv_labels[i]
    temp4 = np.dot(temp3, (np.asmatrix(sv_datapt[i])))
    weights2 += temp4
print("Weights 2",weights2)

#   Calculating b with weights1 and dot product
i = np.random.randint(0,100)
datapt = Input_data[i]
label = labels[i]
b = label - (np.dot(datapt, np.transpose(weights2)))



#   Equation for SVM, using weights1 but can be edited also
print("Equation:", np.transpose(weights2), "*", "x", "+", b, "=1")

line_prod_st = (-weights2[0,0]) * (-0.4)
line_prod2_st = line_prod_st - b
linear_line_st = line_prod2_st / weights2[0,1]

line_prod_end = (-weights2[0,0]) * (1.0)
line_prod2_end = line_prod_end - b
linear_line_end = line_prod2_end / weights2[0,1]

st_x1 = - 0.4
st_x2 = linear_line_st
end_x1 = 1.0
end_x2 = linear_line_end


lx = np.zeros(shape=(2,1))
lx[0] = 0.0
lx[1] = 1.2

ly = np.zeros(shape=(2,1))

ly[0] = (((-weights2[0,0] * (-0.4)) - b) / weights2[0,1])
ly[1] = (((-weights2[0,0] * (1.0)) - b) / weights2[0,1])

#   Graphing data
x1_pos = Input_file["x1"][Input_file["label"] == 1]
x2_pos = Input_file["x2"][Input_file["label"] == 1]
x1_neg = Input_file["x1"][Input_file["label"] == -1]
x2_neg = Input_file["x2"][Input_file["label"] == -1]
plt.scatter(x1_pos,x2_pos,marker='+',c='red', alpha=0.8)
plt.scatter(x1_neg, x2_neg, marker='o', c='blue', alpha=0.8)
plt.scatter(Input_data[above_threshold_indices,0],Input_data[above_threshold_indices,1], marker='*',c='black',alpha=1.0)

plt.plot(lx,ly, "k")
plt.show()


