import numpy as np

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

debug = True

#By Dinh Huan Nguyen
def softmax(z):
    #print("Z: " + str(z))
    z_exp = np.exp(z)
    z_softmax = z_exp / np.sum(z_exp)
    #print('z_softmax:', z_softmax)
    return z_softmax

#By Tim Finnegan X is a n x p numpy array. Y is a n by 1 numpy array
def calculate_loss(model, X, y):
    N = len(y)
    i = 0
    loss = 0
    while (i < 1):
        a = np.dot(X[i,:], model["W1"]) + model["b1"]
        #print('a:', a)
        h = np.tanh(a)
        #print('h:', h)
        z = np.dot(h, model["W2"]) + model["b2"]
        #print('z:', z)
        y_predict = softmax(z)
        #print('y_predict:', y_predict)
        loss += -np.log(y_predict[y[i]])
        #print('loss in:', loss)
        i += 1
    loss = loss / N
    return loss

#By Dinh Huan Nguyen
def predict(model, x):
    #print('x:', x)
    a = np.dot(x, model["W1"]) + model["b1"]
    #print('a:', a)
    h = np.tanh(a)
    #print('h:', h)
    z = np.dot(h, model["W2"]) + model["b2"]
    #print('z:', z)
    y_predict = softmax(z)
    C = 0

    if y_predict[1] > y_predict[0]:
        C = 1

    return C

#By Tim Finnegan. Uses Numpy Arrays. X is a n x p numpy array. Y is a n by 1 numpy array
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    step = 0.01
    model = {
        "W1" : np.random.uniform(-1.0, 1.0,size=(X[0].size, nn_hdim)),
        "W2" : np.random.uniform(-1.0, 1.0,size=(nn_hdim, X[0].size)),
        "b1" : np.random.uniform(-1.0, 1.0, size=(nn_hdim)),
        "b2" : np.random.uniform(-1.0, 1.0, size=(X[0].size))
    }

    for epoch in range(0, num_passes):
        for sample in range(0, X[...,0].size):
            #print("Sample: " + str(X[sample]))
            a = np.dot(X[sample] , model["W1"]) + model["b1"]
            #print("a:" + str(a))
            h = np.tanh(a)
            #print("h: "  + str(h))
            z = np.dot(h, model["W2"]) + model["b2"]
            #print("z: " + str(z))
            y_sam = y[sample]
            if y_sam == 0:
                y_sam = np.array([1,0])
            else:
                y_sam = np.array([0,1])
            #print('y_sam:',str(y_sam))
            y_pred = softmax(z)
            #print('y_pred:',str(y_pred))
            dLdy = y_pred - y_sam
            #print("dLdy: " + str(dLdy))
            dLda = np.multiply(1 - np.multiply(np.tanh(a), np.tanh(a)), np.dot(dLdy , np.transpose(model["W2"])))
            #print("dLda: " + str(dLda))
            dLdW2 = np.dot(np.transpose(np.array([h])), np.array([dLdy]))
            #print("dLdW2: " + str(dLdW2))

            dLdb2 = dLdy
            #print("dLdb2: " + str(X[sample]))
            dLdW1 = np.dot(np.transpose(np.array([X[sample]])), np.array([dLda]))
            #print("dLdW1: " + str(dLdW1))

            dLdb1 = dLda
            #print("dLdb1: " + str(dLdb1))


            model["W1"] = model["W1"] - step * dLdW1
            #print("W1: " + str(model["W1"]))
            model["W2"] = model["W2"] - step * dLdW2
            #print("W2: " + str(model["W2"]))

            model["b1"] = model["b1"] - step * dLdb1
            #print("b1: " + str(model["b1"]))

            model["b2"] = model["b2"] - step * dLdb2
            #print("b2: " + str(model["b2"]))


            if(print_loss):
                print("Loss at Epoch " + str(epoch) + ": " + str(calculate_loss(model, X, y)))
            #print("Epoch: " + str(epoch))
    return model

