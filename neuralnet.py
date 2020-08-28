import numpy as np
import pandas as pd
import cv2
np.random.seed(0)
Input_file = pd.read_csv('downgesture_train.list', header=None, skipinitialspace=True)

weights_L0 = (np.random.uniform(low=-1, high=0, size=(961, 100)))
weights_L1 = (np.random.uniform(low=-1, high=0, size=(101,1)))
#print(weights_L0,weights_L1)
cols = ['name']
Input_file.columns = cols
labels = []
images = []
for i in range(len(Input_file)):
    if ("down") in Input_file["name"][i]:
        labels.append(1)
    else:
        labels.append(0)
Input_file['label'] = labels

for i in range(len(Input_file)):
    im = cv2.imread(Input_file["name"][i].strip(), -1)
    im = im / float(32)
    im = cv2.vconcat(im,im)
    images.append(im)
    images[i] = np.append([1.0], images[i])

def sigmoid(x):
    sigmoid_calc = 1 / (1 + np.exp(np.array(-x)))
    return sigmoid_calc

def error_term_derivative(xi, yi):
    diff = 2 * (np.array(xi) - np.array(yi))
    return diff

def sigmoid_derivative(xi):
    x = np.array(xi) * (1 - np.array(xi))
    #x = 1 - (np.array(xi) ** 2)
    return x


def feedforward(instance, weights0, weights1):
    # Calculating summations for perceptrons layer 1
    sigmoids_L1 = []
    s0 = np.dot(np.transpose(weights0), instance)
    # Calculating sigmoids for perceptron layer 1
    for i in range(len(s0)):
        sigmoids_L1.append(sigmoid(s0[i]))
    # Calculating summations for perceptron layer 2
    sigmoids_L1 = np.append([1.0], sigmoids_L1)
    sigmoids_L2 = []
    s1 = np.dot(np.transpose(weights1), sigmoids_L1)
    # Calculating sigmoids for perceptron layer 2
    for i in range(len(s1)):
        sigmoids_L2.append(sigmoid(s1[i]))
    s0 = np.append([1.0], s0)
    return sigmoids_L1, sigmoids_L2

def backprop(instance, w0,w1,lab=labels):
    test_case = np.transpose(np.asmatrix(instance))
    sigmoid_derivatives = []
    avg_error = []
    weights_bp_L1 = w1[1:]
    xi_L1, xi_L2 = feedforward(instance, weights0=w0, weights1=w1)
    xi_L1_prime = xi_L1[1:]
    delta_L2 = error_term_derivative(xi_L2, labels[0]) * sigmoid_derivative(xi_L2)
    delta_L1 = []
    temp = (weights_bp_L1 * delta_L2)
    temp = np.asmatrix(temp)
    for i in range(len(xi_L1_prime)):
        sigmoid_derivatives.append(sigmoid_derivative(xi_L1_prime[i]))
    sigmoid_derivatives=np.transpose(np.asmatrix(sigmoid_derivatives))
    for i in range(len(temp)):
        delta_L1.append(sigmoid_derivatives[i] * temp[i])
    delta_L1 = np.asmatrix(np.array(delta_L1))
    for i in range(len(xi_L2)):
        avg_error.append(np.average((xi_L2[0]) - lab[i]))
    xi_L1_prime = np.transpose(np.asmatrix(xi_L1))
    delta_L2 = np.transpose(np.asmatrix(delta_L2))
    w1 = w1 - (0.1 * (np.dot(xi_L1_prime, delta_L2)))
    w0 = w0 - (0.1 * (np.dot(test_case, delta_L1)))
    return avg_error, w0, w1 #, np.shape(delta_L1), np.shape(delta_L2), np.shape(sigmoid_derivatives), np.shape(temp)

def train(images, labels_tr):
    weights_L0 = (np.random.uniform(low=-0.1, high=0.1, size=(961, 100)))
    weights_L1 = (np.random.uniform(low=-0.01, high=0.01, size=(101, 1)))
    print(weights_L0,weights_L1)
    iters = 0
    while(iters < 2):
        print("Epoch:", iters)
        iters += 1
        for i in range(len(images)):
            error, weights_L0, weights_L1 = backprop(instance=images[i], w0=weights_L0, w1=weights_L1)

    return weights_L0, weights_L1

wts00, wts12 = train(images, labels)
#print(backprop(instance=images[0], w0=weights_L0, w1=weights_L1))
print(wts00,wts12)
# Score on Training Set

thresh = [0.5]
temp = []
t = 0
for i in range(len(labels)):
    t = feedforward(images[i], weights0=wts00, weights1=wts12)[1]
    print(t)
    if t > thresh:
        t = 1
    else:
        t = 0
    temp.append(t)
pos_count = 0
neg_count = 0

for i in range(len(labels)):
    if (labels[i] == temp[i]):
        pos_count += 1
    else:
        neg_count += 1
score = ((pos_count) / (len(temp))) * 100
print("Training set accuracy:", score)


# Creating Testing Set

Input_file_test = pd.read_csv('downgesture_test.list', header=None, skipinitialspace=True)


cols_test = ['name']
Input_file_test.columns = cols_test
labels_test = []
images_test = []
for i in range(len(Input_file_test)):
    if ("down") in Input_file_test["name"][i]:
        labels_test.append(1)
    else:
        labels_test.append(0)
Input_file_test['label'] = labels_test
for i in range(len(Input_file_test)):
    im = cv2.imread(Input_file_test["name"][i].strip(), -1)
    im = im / float(32)
    im = cv2.vconcat(im,im)
    images_test.append(im)
    images_test[i] = np.append([1.0], images_test[i])


# Scoring on Test set

threshold = [0.5]
pos_count_test = 0
neg_count_test = 0
curr_label_test = []
t2 = 0

for i in range(len(labels_test)):
    t2 = feedforward(images_test[i], weights0=wts00, weights1=wts12)[1]
    if t2 > threshold:
        t2 = 1
    else:
        t2 = 0
    curr_label_test.append(t2)


for i in range(len(labels_test)):
    if (labels_test[i] == curr_label_test[i]):
        pos_count_test += 1
    else:
        neg_count_test += 1
score_test = ((pos_count_test) / (len(labels_test))) * 100
print("Testing set accuracy:", score_test)
