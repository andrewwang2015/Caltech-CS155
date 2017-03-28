import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def loadData(fileName):
    '''
    Takes filename string and returns raw data as numpy array
    '''
    
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    return np.asarray(data[1:], dtype = float)

def getInputsAndOutputs(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y)
    '''    
    inputs = []
    outputs = []
    arr1 = np.asarray([1.0])
    for i in arr:
        inputs.append(np.concatenate((arr1, i[:-1]), axis = 0))                # the x vector
        outputs.append(i[(len(i) - 1)])      # last element, y
    return np.asarray(inputs), np.asarray(outputs)

def returnLoss(predicted, actual):
    ''' 
    Takes two numpy arrays and returns the loss between the two arrays 
    according to squared error
    '''
    
    totalLoss = 0
    assert(len(predicted) == len(actual))
    for i in range(len(predicted)):
        totalLoss += (predicted[i] - actual[i]) ** 2
    return totalLoss

def getPredictionArr(weights, inputs):
    '''
    Takes a weight vector and an inputs array and returns a predicted array of outputs
    '''
    arr = []
    for i in inputs:
        arr.append(np.inner(weights, i))
    return np.asarray(arr)

def calculateGradient(inputData, output, weights):
    '''
    Calculates gradient between an input data and its ouput 
    '''
    assert(len(weights) == len(inputData))
    currentGradient = -2 * (output - np.inner(weights, inputData)) * inputData
    return currentGradient

def oneStepLearning(data):
    '''
    This is the closed-form solution solver
    '''
    inputs, outputs = getInputsAndOutputs(data)
    inputs = np.matrix(inputs)
    outputs = np.matrix(outputs)
    return np.linalg.inv(inputs.transpose() * inputs) * inputs.transpose() * outputs.transpose()
        
        
def runSGD(data, initialWeights, stepSize):
    '''
    Main function that actually runs the full SGD
    '''
    numberEpochs = 0
    allLoss = []
    weights = initialWeights
    inputs, outputs = getInputsAndOutputs(data) 
    initialLoss = returnLoss(getPredictionArr(weights, inputs),outputs)
    allLoss.append(initialLoss)
    while (True):
        np.random.shuffle(data)   #Shuffle data 
        inputs, outputs = getInputsAndOutputs(data) 
        assert(len(inputs) == 1000)
        
        #For all 1000 points ...
        for i in range(len(inputs)):
            gradient = calculateGradient(inputs[i], outputs[i], weights)
            assert(len(gradient) == len(weights))
            weights -= stepSize * gradient    #Update of weights
        numberEpochs += 1
        
        #Return loss after the updating of weights
        currentLoss = returnLoss(getPredictionArr(weights,inputs), outputs)
        
        #Append the loss to a list that keeps track of losses per epoch
        allLoss.append(currentLoss)
        if (math.fabs(allLoss[-1] - allLoss[-2]) / math.fabs (allLoss[1] - allLoss[0]) <= 0.0001):  #Stopping condition
            break
    return weights, numberEpochs, allLoss

def main():
    weights = []
    epochs = []
    loss = []
    possibleStepSizes = [math.exp(-10), math.exp(-11), math.exp(-12), math.exp(-13), math.exp(-14), math.exp(-15)]
    stepSize = math.exp(-15)
    allData = loadData("sgd_data.csv") 
    
    initial = [0.001, 0.001, 0.001, 0.001, 0.001] 
    finalWeights, numEpochs, allLoss = runSGD(allData, initial, stepSize)
    print(finalWeights)   #Answer to 4c
    for i in possibleStepSizes:
        initial = [0.001, 0.001, 0.001, 0.001, 0.001]   #w_o = 0 to incorporate bias term
        finalWeights, numEpochs, allLoss = runSGD(allData, initial, i)
        weights.append(finalWeights)
        epochs.append(numEpochs)
        loss.append(allLoss)
    
    
    #Time to plot
    x = list(range(max(epochs) + 1))
    
    fig = plt.figure()
    plt.title('Training Error vs. Epochs', fontsize = 22)
    plt.plot(list(range(epochs[0] + 1)), loss[0], list(range(epochs[1] + 1)), loss[1],
             list(range(epochs[2] + 1)), loss[2], list(range(epochs[3] + 1)), loss[3],
             list(range(epochs[4] + 1)), loss[4], list(range(epochs[5] + 1)), loss[5], marker = '.')
    plt.legend(('e$^{-10}$ rate', 'e$^{-11}$ rate', 'e$^{-12}$ rate', 'e$^{-13}$ rate', 'e$^{-14}$ rate', 'e$^{-15}$ rate'), loc = 'upper right', fontsize = 14)
    plt.xlabel('Number of Epochs (log scale)', fontsize = 18)
    plt.xscale('log')
    plt.ylabel('Training Error', fontsize = 18)
    plt.margins(y=0.02)
    
    #Answer to 4e
    print(list(oneStepLearning(allData)))
    
main()