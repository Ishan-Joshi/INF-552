import numpy as np
import pandas as pd

#np.random.seed(0)
#   Input data
Input_File = pd.read_csv("clusters.txt", skipinitialspace=True)
columns = ["x1", "x2"]
Input_File.columns=columns
vec_row = []
for i in range(len(Input_File)):
    vec_row.append([Input_File["x1"][i], Input_File["x2"][i]])
vec_data = np.array(vec_row)
vec_data = np.asmatrix(vec_data)
vec_data = np.transpose(vec_data)
datapoint_weights = np.random.rand(len(Input_File), 3)

#   Normalizing Weights (ric)
for i in range(len(Input_File)):
    sum = np.sum(datapoint_weights[i])
    for j in range(0,3):
        datapoint_weights[i,j] = datapoint_weights[i, j] / sum

#   E and M step
def fit():
    iterations = 0
    old_likelihood = 0.0
    flag = False
    while(flag == False):
        iterations += 1
        #   Pi's
        product = 0
        pi = []
        for k in range(0,3):
            pi.append(Calc_Pi(vec_data, datapoint_weights[:,k]))
        #   M step
        averages = []
        covmat = []
        for i in range(0,3):
            averages.append(Gauss_parameters(vec_data, datapoint_weights[:,i])[0])
            covmat.append(Gauss_parameters(vec_data, datapoint_weights[:,i])[1])

        #   Likelihood logarithms
        likelihood = 0.0
        for i in range(vec_data.shape[1]):
            sub_likelihood = []
            for k in range(0, 3):
                sub_likelihood.append(pi[k] * Gauss_PDF(vec_data[:, i], averages[k], covmat[k]))
            sum_likelihood = np.sum(sub_likelihood)
            likelihood += np.log(sum_likelihood)
        print("Iterations:{0}, Likelihood: {1}".format(iterations, likelihood))
        diff = likelihood - old_likelihood
        if (old_likelihood != 0 and (diff) < 0.01):
            print("Threshold Reached")
            flag = True
            break
        old_likelihood = likelihood

        # E step

        for i in range(vec_data.shape[1]):
            pdfs = []
            for k in range(0,3):
                pdfs.append(pi[k] * (Gauss_PDF(vec_data[:,i], averages[k], covmat[k])))
            sum_pdfs = np.sum(pdfs)
            pdfs = pdfs / sum_pdfs
            for j in range(0,3):
                datapoint_weights[i,j] = pdfs[j] / sum_pdfs
            for i in range(vec_data.shape[1]):
                sum2 = np.sum(datapoint_weights[i])
                for j in range(0, 3):
                    datapoint_weights[i, j] = datapoint_weights[i, j] / sum2


    return averages, covmat, pi


def Gauss_parameters(datapoints, cluster_weights):
    #   Weighted Average
    weighted_average = 0
    for i in range(datapoints.shape[1]):
        prod = datapoints[:,i] * cluster_weights[i]
        weighted_average += prod
    weighted_average = weighted_average / np.sum(cluster_weights)

    #   Covariance Matrix
    covariance_product = 0
    for i in range(datapoints.shape[1]):
        normalized_matrix = datapoints[:, i] - weighted_average
        #covariance_product += cluster_weights[i] * (np.dot(normalized_matrix, np.transpose(normalized_matrix)))
        covariance_product += cluster_weights[i] * ((normalized_matrix * np.transpose(normalized_matrix)))
    covariance_matrix = covariance_product / np.sum(cluster_weights)
    return weighted_average, covariance_matrix

def Calc_Pi(datapoints, cluster_weights):
    # Pi
    pi_sum = 0
    for i in range(datapoints.shape[1]):
        pi_sum += cluster_weights[i]
    pi = pi_sum / datapoints.shape[1]
    return pi

def Gauss_PDF(datapoint, weighted_average, covariance_matrix):
    #weighted_average, covariance_matrix = Gauss_parameters(vec_data, cluster_weights)
    norm = datapoint - weighted_average
    exponent = (np.transpose(norm)) * (np.linalg.inv(covariance_matrix)) * (norm)
    denominator = 1 / ((2*np.pi) * (np.linalg.det(covariance_matrix) ** 0.5))
    distribution = denominator * (np.power((np.e), (-0.5 * exponent)))
    return distribution
print(fit())

