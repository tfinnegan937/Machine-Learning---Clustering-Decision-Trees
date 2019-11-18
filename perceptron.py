import numpy as np


def perceptron_train(X, Y):
    w = np.zeros(X[0].size)
    b = 0
    update = True
    count = 0
    while update and count < 10000:
        update = False
        for sample in range(0, X[...,0].size):
            a = np.dot(w, X[sample]) + b
            if Y[sample][0] * a <= 0:
                update = True
                w = w + (X[sample] * Y[sample][0])
                b = b + Y[sample][0]
        count = count + 1

    return w, b

def perceptron_test(X_test, Y_test, w, b):
    sum = 0
    num_elements = Y_test[...,0].size

    for sample in range(0, num_elements):
        a = np.dot(w, X_test[sample]) + b
        if a >= 0:
            y_prediction = 1
        else:
            y_prediction = -1
        if y_prediction == Y_test[sample][0]:
            sum = sum + 1

    return sum / num_elements
