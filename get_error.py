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


weights12 = np.array([2,2,2,2,2,2]).reshape(2,3)
weights23 = np.array([1,1]).reshape(1,2)

x = np.array([0,1,2]).reshape(3,1)
y = np.array([1]).reshape(1,1)

def delta_y(nn_out):
    return (y - nn_out)*sigmoid_prime(nn_out)


nn_sums_2 = weights12.dot(x)
nn_out_2 = sigmoid(nn_sums_2)

nn_sums_3 = weights23.dot(nn_out_2)
nn_out_3 = sigmoid(nn_sums_3)


delta_3 = delta_y(nn_out_3)

delta_2 = get_error(delta_3,nn_sums_3,weights23)

delta_1 = get_error(delta_2.T,nn_sums_2, weights12)


def dJ_dw_2(j,k):
    return(delta_2[j,0]*x[k,0])

print(dJ_dw_2(0, 2))
print(dJ_dw_2(1, 2))
