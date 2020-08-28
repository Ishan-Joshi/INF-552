import numpy as np
import pandas as pd
#np.random.seed(0)
Input_file = pd.read_csv('classification.txt', header=None, skipinitialspace=True, sep=',')
Input_file = Input_file.iloc[:,[0,1,2,4]]
cols = ['x1', 'x2', 'x3', 'y']
Input_file.columns = cols
Input_file_array = np.array(Input_file)
labels = np.transpose(np.asmatrix(Input_file["y"]))
cntpos = 0
cntneg = 0
for i in range(len(labels)):
    if (labels[i] == 1):
        cntpos += 1
    else:
        cntneg += 1
print(cntpos, cntneg)
#Converting input features into a matrix
Input_matrix = np.asmatrix(Input_file.iloc[:,0:3])
data = np.zeros((2000, 4))
for i in range(len(Input_file_array)):
    data[i] = (np.append([1.0], Input_file_array[i, 0:3]))
data = np.asmatrix(data)
def logistic_train():
    #Initializing weights
    weights = np.random.uniform(low=-100, high=100,size=(np.shape(Input_matrix)[1]+1))
    # Below are the various trials used for hyperparameter tuning of weights:
    # [-0.03086503, -0.17809694,  0.11405758, 0.07627853]
    #  np.random.uniform(low=-10, high=10,size=(np.shape(Input_matrix)[1]+1))
    # np.random.uniform(low=-0.5, high=0.5,size=(np.shape(Input_matrix)[1]+1))
    # [-0.03150075, -0.17769619, 0.11445235, 0.07670126]
    #[ 0.00506366, -0.08203579, -0.02436907, -0.03668983]
    # [ 0.0692835, -0.07307421, 0.04851551, 0.09105342]
    # np.random.random(np.shape(Input_matrix)[1] + 1)
    weights = np.asmatrix(weights)
    print("Initial weights{0}".format(weights))
    predicted_output_prob = np.zeros((len(labels), 1))
    iters = 0
    while (iters < 7000):
        gradient_error = 0
        neg = 0
        pos = 0
        #tr = 0
        iters += 1
        for i in range(len(labels)):
            predicted_output_prob[i] = predict(data[i], weights, labels[i])[0]
            if (predicted_output_prob[i] > 0.5):
                predicted_output_prob[i] = 1
                pos += 1
            else:
                predicted_output_prob[i] = -1
                neg += 1
        for i in range(len(predicted_output_prob)):
            grad = predict(data[i], weights, labels[i])[1]
            gradient_error += grad
        gradient_error = (-gradient_error) / len(data)
        print("Iteration:{0}".format(iters))
        weights = weights - (0.1 * gradient_error )
    print("Final weights{0}".format(weights))
    return predicted_output_prob

def predict(datapoint, weights, label):
    exp_one = np.dot(datapoint, np.transpose(weights))
    exp_final = label * exp_one
    sigmoid = (np.exp(exp_final)) / (1 + (np.exp(exp_final)))
    gradient = (label * datapoint) / (1 + (np.exp(exp_final)))
    return sigmoid, gradient

score = 0
predicted_label = logistic_train()
false_pos = 0
false_neg = 0
true_val = 0
cnt3 = 0
true_val = np.where(labels==predicted_label)[0].shape[0]
score = true_val / len(labels)
print("Accuracy is {0}".format(score * 100))
