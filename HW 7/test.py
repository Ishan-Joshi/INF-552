import numpy as np
import pandas as pd
#   For X1
P_X1 = np.zeros((10,2))
t = 0
for i in range(np.shape(P_X1)[0]):
    t = i + 1
    P_X1[i] = [t, 0.1]
P_X1_vals = P_X1[:,1]


P_E1_given_X1 = np.zeros((12,10))
temp = 0

for j in range(np.shape(P_E1_given_X1)[1]):
    for i in range(np.shape(P_E1_given_X1)[0]):
        temp = j + 1
        if (i == temp - 1):
            P_E1_given_X1[i,j] = 0.33
        elif (i == temp):
            P_E1_given_X1[i,j] = 0.33
        elif(i == temp + 1):
            P_E1_given_X1[i,j] = 0.33
        else:
            P_E1_given_X1[i,j] = 0.0
P_E1X1_times_P_X1 = np.zeros((12,10))
for i in range(len(P_X1)):
    for j in range(len(P_E1_given_X1)):
        P_E1X1_times_P_X1[j,i] = P_E1_given_X1[j,i] * P_X1_vals[i]
proj_mat1 = np.zeros((10,3))
for i in range(len(proj_mat1)):
    proj_mat1[i] = [i+1, np.argmax(P_E1X1_times_P_X1[:,i]), np.max(P_E1X1_times_P_X1[:,i])]
#print(proj_mat1)
P_X1_post_projection = proj_mat1[:,2]
#print(P_X1_post_projection)
"""def multiply_xt_xtprev():
    P_xt_xtprev = np.zeros((10,10))
    for j in range(0,10):
        temp3 = j + 1
        if (temp3 == 1):
            P_xt_xtprev[1,j] = 1
        elif (temp3 == 10):
            P_xt_xtprev[8,j] = 1
        else:
            temp1 = j + 1
            temp2 = j - 1
            P_xt_xtprev[temp2, j] = 0.5
            P_xt_xtprev[temp1, j] = 0.5
    return P_xt_xtprev

def projectXt(product_Pxt_xtprev_and_Pxt):
    P_project = np.zeros((10,3))
    for j in range(0,10):
        P_project[j] =  [j+1,np.argmax(product_Pxt_xtprev_and_Pxt[:,j]), np.max(product_Pxt_xtprev_and_Pxt[:,j])]
    return P_project

def multiply_Xt_and_Et():
    P_Et_cond_Xt = np.zeros((12,10))
    for j in range(0,10):
        for i in range(0,12):
            temp = j + 1
            if (i == temp - 1):
                P_Et_cond_Xt[i, j] = 0.33
            elif (i == temp):
                P_Et_cond_Xt[i, j] = 0.33
            elif (i == temp + 1):
                P_Et_cond_Xt[i, j] = 0.33
            else:
                P_Et_cond_Xt[i, j] = 0.0
    return P_Et_cond_Xt

def main_func():
    #   Find P(Xt/Xt-1)
    P_Xt_Xtprev = multiply_xt_xtprev()
    #   Find P(Xt/Xt-1)*P(X1)
    product_Pxt_xtprev_and_Pxt = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            product_Pxt_xtprev_and_Pxt[j,i] = P_Xt_Xtprev[j,i] * P_X1_post_projection[i]
    P_project = projectXt(product_Pxt_xtprev_and_Pxt)
    P_project_val = P_project[:,2]
    P_Etcond_Xt = multiply_Xt_and_Et()
    product_PXt_PEtcondXt = np.zeros((12,10))
    for i in range(10):
        for j in range(12):
            product_PXt_PEtcondXt[j,i] = P_Etcond_Xt[j,i] * P_project_val[i]
    P_project_Et_on_Xt = projectXt(product_PXt_PEtcondXt)
    return product_PXt_PEtcondXt
print(main_func())"""

print(P_E1_given_X1[7] * P_X1_vals)
print(P_E1_given_X1[6])

#   X2 and X1
"""P_xt_xtprev = np.zeros((10,10))
for j in range(0,10):
    temp3 = j + 1
    if (temp3 == 1):
        P_xt_xtprev[1,j] = 1
    elif (temp3 == 10):
        P_xt_xtprev[8,j] = 1
    else:
        temp1 = j + 1
        temp2 = j - 1
        P_xt_xtprev[temp2, j] = 0.5
        P_xt_xtprev[temp1, j] = 0.5
#print(np.shape(P_xt_xtprev[]))

P_X2 = np.zeros((10,1))

for i in range(len(P_xt_xtprev)):
    #P_temp = np.zeros((10, 1))
    P_temp = P_xt_xtprev[i] * P_E1_X1
    P_X2[i] = np.max(P_temp)
#print(np.transpose(P_X2))
#   E2/X2
P_Et_cond_Xt = np.zeros((12,10))
for j in range(0,10):
    for i in range(0,12):
        temp = j + 1
        if (i == temp - 1):
            P_Et_cond_Xt[i, j] = 0.33
        elif (i == temp):
            P_Et_cond_Xt[i, j] = 0.33
        elif (i == temp + 1):
            P_Et_cond_Xt[i, j] = 0.33
        else:
            P_Et_cond_Xt[i, j] = 0.0
#print(P_Et_cond_Xt[6])
P_Et_Xt = P_Et_cond_Xt[observation_sequence[1]] * np.transpose(P_X2)
#print(P_Et_Xt)
predicted_sequence[t] = [np.argmax(P_Et_Xt)+1, np.max(P_Et_Xt)]
print(predicted_sequence)

# X3 and X2
P_xt_xtprev = np.zeros((10,10))
for j in range(0,10):
    temp3 = j + 1
    if (temp3 == 1):
        P_xt_xtprev[1,j] = 1
    elif (temp3 == 10):
        P_xt_xtprev[8,j] = 1
    else:
        temp1 = j + 1
        temp2 = j - 1
        P_xt_xtprev[temp2, j] = 0.5
        P_xt_xtprev[temp1, j] = 0.5
#print(np.shape(P_xt_xtprev[]))

P_X2 = np.zeros((10,1))

for i in range(len(P_xt_xtprev)):
    #P_temp = np.zeros((10, 1))
    P_temp = P_xt_xtprev[i] * P_Et_Xt
    P_X2[i] = np.max(P_temp)
#print(np.transpose(P_X2))
#   E2/X2
P_Et_cond_Xt = np.zeros((12,10))
for j in range(0,10):
    for i in range(0,12):
        temp = j + 1
        if (i == temp - 1):
            P_Et_cond_Xt[i, j] = 0.33
        elif (i == temp):
            P_Et_cond_Xt[i, j] = 0.33
        elif (i == temp + 1):
            P_Et_cond_Xt[i, j] = 0.33
        else:
            P_Et_cond_Xt[i, j] = 0.0
#print(P_Et_cond_Xt[6])
P_Et_Xt2 = P_Et_cond_Xt[observation_sequence[2]] * np.transpose(P_X2)
#print(P_Et_Xt)
predicted_sequence[2] = [np.argmax(P_Et_Xt2)+1, np.max(P_Et_Xt2)]
print(predicted_sequence)"""
