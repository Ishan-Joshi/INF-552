import numpy as np

observation_sequence = [8, 6, 4, 6, 5, 4, 5, 5, 7, 9]
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
P_E1_X1 = P_E1_given_X1[observation_sequence[0]] * P_X1_vals
predicted_sequence = np.zeros((10,2))
predicted_sequence[0] = [np.argmax(P_E1_X1)+1, np.max(P_E1_X1)]



def multiply_xt_xtprev():
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

def main_function():
    #   Xt and Xt-1
    P_Et_Xt = np.zeros((1, 10))
    for temp in range(1,10):
        P_xt_xt_prev = multiply_xt_xtprev()
        P_Xti = np.zeros((10, 1))

        for i in range(len(P_xt_xt_prev)):
            if (temp == 1):
                P_temp = P_xt_xt_prev[i] * P_E1_X1
            else:
                P_temp = P_xt_xt_prev[i] * P_Et_Xt
            P_Xti[i] = np.max(P_temp)
        PEt_Xt = multiply_Xt_and_Et()
        P_Et_Xt = PEt_Xt[observation_sequence[temp]] * np.transpose(P_Xti)
        predicted_sequence[temp] = [np.argmax(P_Et_Xt) + 1, np.max(P_Et_Xt)]
    print(predicted_sequence[:,0])
print(main_function())