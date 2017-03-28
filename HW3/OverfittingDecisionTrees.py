import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy
from sklearn.tree import DecisionTreeClassifier

def loadData(fileName):
    '''
    Takes filename string and returns raw data as numpy array
    '''
    
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    return np.asarray(data)

def deleteColumn(column, arr):
    '''
    Deletes particular column in an array 
    '''
    arr1 = scipy.delete(arr, column, 1)
    return arr1

def getInputsAndOutputsTraditional(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y) without normalization
    '''    
    inputs = []
    outputs = []
    arr1 = np.asarray([1.0])
    for i in arr:
        inputs.append(i[1:])                # the x vector
        if i[0] == 'M':   # 'M' corresponds to 1
            outputs.append(1)      # first, y
        else:
            outputs.append(0)   # 'B' corresponds to 0
    return np.asarray(inputs, dtype = float), np.asarray(outputs, dtype = float)

def returnLoss(predicted, actual):
    ''' 
    Takes two numpy arrays and returns the loss between the two arrays 
    according to 0/1 loss
    '''
    
    totalLoss = 0
    assert(len(predicted) == len(actual))
    for i in range(len(predicted)):
        if predicted[i] != actual[i]:
            totalLoss += 1
    return totalLoss / len(predicted)

def runDecisionTreeNodeSize(nodeSize, trainingInputs, trainingOutputs, testInputs, testOutputs):
    '''
    This function runs the decision tree based on min. leaf size 
    '''
    trainingPredicted = []
    testingPredicted = []
    clf = DecisionTreeClassifier(min_samples_leaf = nodeSize)
    clf.fit(trainingInputs, trainingOutputs)
    trainingPredicted = clf.predict(trainingInputs)
    testingPredicted = clf.predict(testInputs)
    trainingError = returnLoss(trainingPredicted, trainingOutputs)
    testingError = returnLoss(testingPredicted, testOutputs)
    return trainingError, testingError
        
        
def runDecisionTreeNodeDepth(depth, trainingInputs, trainingOutputs, testInputs, testOutputs):
    '''
    This function runs the decision tree based on max. depth 
    '''
    trainingPredicted = []
    testingPredicted = []
    clf = DecisionTreeClassifier(max_depth = depth)
    clf.fit(trainingInputs, trainingOutputs)
    trainingPredicted = clf.predict(trainingInputs)
    testingPredicted = clf.predict(testInputs)
    trainingError = returnLoss(trainingPredicted, trainingOutputs)
    testingError = returnLoss(testingPredicted, testOutputs)
    return trainingError, testingError    
    
def main():
    allData = loadData("wdbc.data.csv")
    allData = deleteColumn(0, allData)
    training = allData[:400]
    testing = allData[400:]
    trainingIn, trainingOut = getInputsAndOutputsTraditional(training)
    testingIn, testingOut = getInputsAndOutputsTraditional(testing)
    
    #leaf node size
    leafNodesTrainingError = []
    leafNodesTestingError = []
    xRange = []
    for i in range(1, 26):
        x, y = runDecisionTreeNodeSize(i, trainingIn, trainingOut, testingIn, testingOut)
        leafNodesTrainingError.append(x)
        leafNodesTestingError.append(y)
        xRange.append(i)
    

    
    fig = plt.figure()
    plt.title('Error vs. Leaf Node Size', fontsize = 22)    
    plt.plot(xRange, leafNodesTrainingError, xRange, leafNodesTestingError, marker = '.', linewidth = 2)
    plt.legend(('Training', 'Testing'), loc = 'best', fontsize = 14)
    plt.xlabel('Leaf Node Size', fontsize = 18)
    plt.ylabel('Error', fontsize = 18)
    plt.margins(y=0.02)     
    
#max depth size
    leafNodesTrainingError = []
    leafNodesTestingError = []
    xRange = []
    for i in range(2, 21):
        x, y = runDecisionTreeNodeDepth(i, trainingIn, trainingOut, testingIn, testingOut)
        leafNodesTrainingError.append(x)
        leafNodesTestingError.append(y)
        xRange.append(i)
        
    fig = plt.figure()
    plt.title('Error vs. Max. Tree Depth', fontsize = 22)    
    plt.plot(xRange, leafNodesTrainingError, xRange, leafNodesTestingError, marker = '.', linewidth = 2)
    plt.legend(('Training', 'Testing'), loc = 'best', fontsize = 14)
    plt.xlabel('Max. Tree Depth', fontsize = 18)
    plt.ylabel('Error', fontsize = 18)
    plt.margins(y=0.02)             
main ()