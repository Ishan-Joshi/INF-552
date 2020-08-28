import numpy as np
import pandas as pd
#np.random.seed(0)
Input_file = pd.read_csv('classification.txt', header=None, skipinitialspace=True, sep=',')
Input_file = Input_file.iloc[:,0:4]
cols = ['x1', 'x2', 'x3', 'y']
Input_file.columns = cols
Input_file_array = np.array(Input_file)
labels = np.transpose(np.asmatrix(Input_file["y"]))
#Converting input features into a matrix
Input_matrix = np.asmatrix(Input_file.iloc[:,0:3])
data = np.zeros((2000, 4))
for i in range(len(Input_file_array)):
    data[i] = (np.append([1.0], Input_file_array[i, 0:3]))
data = np.asmatrix(data)

def perceptron_train():
    #Initializing weights
    weights = np.random.uniform(low= -100, high= 100, size=(np.shape(Input_matrix)[1]+1))
    # Below are the various hyperparameter tuning options used to obtain optimal results:
    # np.random.random(np.shape(Input_matrix)[1] + 1)
    # uniform(low= -10, high= 10, size=(np.shape(Input_matrix)[1]+1))
    weights = np.asmatrix(weights)
    print("Initial weights{0}".format(weights))
    predicted_output_labels = np.zeros((len(labels), 1))
    iters = 0
    flag = 0
    while (iters < 10000):
        iters += 1
        for i in range(len(labels)):
            predicted_output_labels[i] = predict(data[i], weights, labels[i])
        error = (np.zeros((len(data), 1)))
        error = np.asmatrix(error)
        error = labels - predicted_output_labels
        false_pos = 0
        false_neg = 0
        true_val = 0
        cnt3 = 0
        for i in range(len(error)):
            if error[i] == -2:
                false_pos += 1
            elif error[i] == 2:
                false_neg += 1
            else:
                true_val += 1
        print("Iteration {0}, True classifications:{1}".format(iters, true_val))
        if (true_val == len(error)):
            flag = 1
            print(flag)
            break
        for i in range(len(error)):
            if (predicted_output_labels[i] > 0) and (labels[i] < 0):
                weights = weights - (0.01 * data[i])
            elif (predicted_output_labels[i] < 0) and (labels[i] > 0):
                weights = weights + (0.01 * data[i])
    print("Final weights{0}".format(weights))
    return predicted_output_labels, weights

def predict(datapoint, updated_weights, label):
    predicted_label = np.dot(datapoint, np.transpose(updated_weights))
    if (predicted_label > 0):
        return 1
    else:
        return -1

score = 0
predicted_label = perceptron_train()[0]
for i in range(len(labels)):
    if labels[i] == predicted_label[i]:
        score += 1
score = score / 2000
print("Accuracy is {0}".format(score * 100))