import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
Input_file = pd.read_csv("fastmap-data.txt", skipinitialspace=True, header=None, sep='[\t]', engine='python')
cols = ["ID1", "ID2", "Symmetric Distance"]
Input_file.columns = cols
data = []
for i in range(len(Input_file)):
    data.append(Input_file.loc[i,:])
iterations = 0
x = np.zeros((11,3))
print()

def Find_Farthest_Pair(iterations):
    a = np.random.randint(1,9)
    b = 0
    for i in range(10):
        curr_max = 0
        for j in range(1, 11):
            dist = get_distance(iterations, a, j)
            if(dist > curr_max):
                curr_max = dist
                b = j

        a = b
    return a, j
def get_distance(iterations, a, b):
    if (iterations == 1) and (a != b):
        return Input_file[Input_file["ID1"] == np.minimum(a,b)][Input_file["ID2"] == np.maximum(a,b)].iloc[0,2]
    elif (iterations >= 1) and (a == b):
        return 0
    else:
        distance = np.power(get_distance(iterations-1, a, b), 2) - (np.power((x[a][iterations-1] - x[b][iterations-1]),2))
        return (distance ** 0.5)



def fastmap(k):
    #print(k)
    global iterations
    if (k < 1):
        print("Recursion limit reached")
        return None


    else:
        iterations += 1
    a,b = Find_Farthest_Pair(iterations)
    if(a == b):
        print("Invalid randomisation. Please run code again.")
        return None
    print("Iteration {}".format(iterations))
    print("A:{}".format(a))
    print("B:{}".format(b))
    for i in range(1,11):
        if (i == a):
            x[i, iterations] = 0
        elif (i == b):
            x[i, iterations] = get_distance(1, a, b)
        else:
            numerator = (get_distance(iterations, a, i) ** 2) + (get_distance(iterations, a, b) ** 2) - (get_distance(iterations, i, b) ** 2)
            denominator = 2 * get_distance(iterations, a, b)
            x[i, iterations] = numerator / denominator
    k = k -1
    fastmap(k)
fastmap(2)


word_list = pd.read_csv("fastmap-wordlist.txt", header=None, skipinitialspace=True)
words = []
for i in range(len(word_list)):
    words.append(word_list.iloc[i,0])




for i in range(1,11):
    plt.scatter(x[i, 1], x[i,2])
    plt.annotate(words[i-1], (x[i,1], x[i,2]))
plt.savefig("Mapped wordlist")
plt.close()