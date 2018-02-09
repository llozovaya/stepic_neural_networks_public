# речь про back propagation
# Задание 3.2 - 10

import numpy as np

## Определим разные полезные функции

def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))

def activation_max(x):
    if x > 0:
        return x
    return 0

def max_prime(x):
    if x > 0:
        return 1
    return 0


def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    # here goes your code

    n = deltas.shape[0]
    return(np.sum(weights.T.dot(deltas.T)*sigmoid_prime(sums.T), axis=1, keepdims=True)/n)


def get_error_strange(deltas, sums,weights):
    n = deltas.shape[0]
    mult = weights.T.dot(deltas.T)
    prime = np.array([[max_prime(sums[0,0])], [sigmoid_prime(sums[1,0])]])
    return(np.sum(mult*prime, axis=1, keepdims=True)/n)

#weights12 = np.array([2,2,2,2,2,2]).reshape(2,3)
#weights23 = np.array([1,1]).reshape(1,2)

#weights12 = np.array([8.0,10.0,7.0,10.0,8.0,9.0]).reshape(2,3)
#weights23 = np.array([10.0, 9.0]).reshape(1,2)

#weights12 = np.array([0.2,0.9,0.6,0.2,0.3,0.7]).reshape(2,3)
#weights23 = np.array([0.2,0.5]).reshape(1,2)

weights12 = np.array([0.7, 0.2, 0.7, 0.8, 0.3, 0.6]).reshape(2,3)
weights23 = np.array([0.2,0.4]).reshape(1,2)

#x = np.array([0,1,2]).reshape(3,1)
#x = np.array([15.0,5.0,15.0]).reshape(3,1)
x = np.array([0.0, 1.0, 1.0]).reshape(3,1)

y = np.array([1]).reshape(1,1)

def delta_y(nn_out):
    return (sigmoid(nn_out) - y)*sigmoid_prime(nn_out)


nn_sums_2 = weights12.dot(x)
#nn_out_2 = sigmoid(nn_sums_2)
nn_out_2 = np.array([[activation_max(nn_sums_2[0,0])], [sigmoid(nn_sums_2[1,0])]])

nn_sums_3 = weights23.dot(nn_out_2)
#nn_out_3 = sigmoid(nn_sums_3)


delta_3 = delta_y(nn_sums_3)

#delta_2 = get_error(delta_3,nn_sums_3,weights23)
delta_2 = get_error_strange(delta_3,nn_sums_2,weights23)

#delta_1 = get_error(delta_2.T,nn_sums_1, weights12)


def dJ_dw_2(j,k):
    return(delta_2[j,0]*x[k,0])

print(delta_2)
print(dJ_dw_2(0, 2))
print(dJ_dw_2(1, 2))
