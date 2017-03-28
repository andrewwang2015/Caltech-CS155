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

def normalizeInputData(inputs):
    inputsNormalized = inputs
    means = []
    stds = []
    for i in range(1, len(inputsNormalized[0])):
        means.append(np.mean(inputsNormalized[:,[i]]))
        stds.append(np.std(inputsNormalized[:,[i]]))
    for i in inputsNormalized:
        for j in range(1, len(i)):
            i[j] = (i[j] - means[j-1]) / stds[j-1]
    return inputsNormalized

def normalizeOutputData(training, testing):
    '''
    Normalizes testing data against training data
    '''
    testingNormalized = testing
    trainingNormalized = training
    means = []
    stds = []
    for i in range(1, len(trainingNormalized[0])):
        means.append(np.mean(trainingNormalized[:,[i]]))
        stds.append(np.std(trainingNormalized[:,[i]])) 
    for i in testingNormalized:
        for j in range(1, len(i)):
            i[j] = (i[j] - means[j-1]) / stds[j-1]
    return testingNormalized
    
def getInputsAndOutputs(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y) with normalization
    '''    
    inputs = []
    outputs = []
    arr1 = np.asarray([1.0])
    for i in arr:
        inputs.append(np.concatenate((arr1, i[1:]), axis = 0))                # the x vector
        outputs.append(i[0])      # first, y
        assert(i[0] == -1 or i[0] == 1)
    return normalizeInputData(np.asarray(inputs)), np.asarray(outputs)

def getInputsAndOutputsTraditional(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y) without normalization
    '''    
    inputs = []
    outputs = []
    arr1 = np.asarray([1.0])
    for i in arr:
        inputs.append(np.concatenate((arr1, i[1:]), axis = 0))                # the x vector
        outputs.append(i[0])      # first, y
        assert(i[0] == -1 or i[0] == 1)
    return np.asarray(inputs), np.asarray(outputs)

def returnRegularizedLoss(weights, outputs, inputs, lambda1):
    totalLoss = 0
    assert(len(inputs) == len(outputs))
    for i in range(len(inputs)):
        if (outputs[i] == -1):
            currentTerm = np.log(1 / (1 + math.exp(np.inner(weights, inputs[i]))))
        else:
            assert(outputs[i] == 1)
            currentTerm = np.log(1 / (1 + math.exp(-np.inner(weights, inputs[i]))))
        totalLoss += currentTerm
    return (-1 * totalLoss + lambda1 * np.inner(weights, weights))  / len(inputs)

def returnLoss(weights, outputs, inputs):
    ''' 
    Takes two numpy arrays and returns the loss between the two arrays 
    according to log loss
    '''
    
    totalLoss = 0
    assert(len(inputs) == len(outputs))
    for i in range(len(inputs)):
        if (outputs[i] == -1):
            currentTerm = np.log(1 / (1 + math.exp(np.inner(weights, inputs[i]))))
        else:
            assert(outputs[i] == 1)
            currentTerm = np.log(1 / (1 + math.exp(-np.inner(weights, inputs[i]))))
        totalLoss += currentTerm
    return -1 * totalLoss / len(inputs)

def calculateGradient(inputData, output, weights, lambda1, size):
    '''
    Calculates gradient between an input data and its output 
    '''
    assert(output == 1 or output == -1)
    assert(len(weights) == len(inputData))
    currentGradient =  2* lambda1 * weights / size - inputData * output / (math.exp(np.inner(weights, inputData) * output) + 1)
    assert(len(currentGradient) == len(inputData))
    return currentGradient

def returnL2Norm(weights):
    return math.sqrt((np.inner(weights, weights)))
    
def runSGD(data, initialWeights, stepSize, lambda1):
    '''
    Main function that actually runs the full SGD
    '''
    allLossRegularized = []
    numberEpochs = 0
    allLoss = []
    weights = initialWeights
    inputs, outputs = getInputsAndOutputs(data) 
    #inputs = normalizeInputData(inputs)
    initialLoss = returnLoss(weights, outputs, inputs)
    allLossRegularized.append(returnRegularizedLoss(weights, outputs, inputs, lambda1))
    allLoss.append(initialLoss)
    while (True):
        np.random.shuffle(data)   #Shuffle data 
        inputs, outputs = getInputsAndOutputs(data) 
        #inputs = normalizeInputData(inputs)
        for i in range(len(inputs)):
            gradient = calculateGradient(inputs[i], outputs[i], weights, lambda1, len(inputs))
            assert(len(gradient) == len(weights))
            weights -= stepSize * gradient    #Update of weights
        numberEpochs += 1
        
        #Return loss after the updating of weights
        currentLoss = returnLoss(weights, outputs, inputs)
        allLossRegularized.append(returnRegularizedLoss(weights, outputs, inputs, lambda1))
        #Append the loss to a list that keeps track of losses per epoch
        allLoss.append(currentLoss)
        if (math.fabs(allLossRegularized[-1] - allLossRegularized[-2]) / math.fabs (allLossRegularized[1] - allLossRegularized[0]) <= 0.0001):  #Stopping condition
            break
    return weights, numberEpochs, allLoss

def main():
    
    #WINE TRAINING 1
    lambdas = []
    weights = []
    epochs = []
    loss = []
    stepSize = math.exp(-4)
    initialLambda = 0.0001
    allData = loadData("wine_training1.txt") 
    for i in range(15):
        initial = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]   #w_o = 0 to incorporate bias term
        initial = np.asarray(initial, dtype = float)        
        finalWeights, numEpochs, allLoss = runSGD(allData, initial, stepSize, initialLambda)
        weights.append(finalWeights)
        epochs.append(numEpochs)
        loss.append(allLoss[-1])
        lambdas.append(initialLambda)
        initialLambda *= 3
        
    # WINE TRAINING 2
    lambdas1 = []
    weights1 = []
    epochs1 = []
    loss1 = []
    stepSize = math.exp(-4)
    initialLambda = 0.0001
    allData1 = loadData("wine_training2.txt") 
    for i in range(15):
        initial = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]   #w_o = 0 to incorporate bias term
        initial = np.asarray(initial, dtype = float)        
        finalWeights1, numEpochs1, allLoss1 = runSGD(allData1, initial, stepSize, initialLambda)
        weights1.append(finalWeights1)
        epochs1.append(numEpochs1)
        loss1.append(allLoss1[-1])
        lambdas1.append(initialLambda)
        initialLambda *= 3

    fig = plt.figure()
    plt.title(r'Training Error vs. $\lambda$', fontsize = 22)    
    plt.plot(lambdas, loss, lambdas1, loss1, marker = '.')
    plt.legend(('Training Set 1', 'Training Set 2'), loc = 'best', fontsize = 14)
    plt.xscale('log')
    plt.xlabel('$\lambda$ (log scale)', fontsize = 18)
    plt.ylabel('Training Error', fontsize = 18)
    plt.margins(y=0.02)
    
 

    
    # TEST SET
    trainingInputs1, trainingOutputs1 = getInputsAndOutputsTraditional(allData)
    trainingInputs2, trainingOutputs2 = getInputsAndOutputsTraditional(allData1)
    testData = loadData("wine_testing.txt") 
    testInputs1, testOutputs1 = getInputsAndOutputsTraditional(testData)
    testInputs2, testOutputs2 = getInputsAndOutputsTraditional(testData)
    testInputs1NormalizedSet1 = normalizeOutputData(trainingInputs1, testInputs1)
    
    testInputs1NormalizedSet2 = normalizeOutputData(trainingInputs2, testInputs2)
    #print(testInputs1NormalizedSet2)
    set1TestErrors = []
    set2TestErrors = []
    for i in weights:
        set1TestErrors.append(returnLoss(i, testOutputs1, testInputs1NormalizedSet1))
    for j in weights1:
        set2TestErrors.append(returnLoss(j, testOutputs1, testInputs1NormalizedSet2))
    
    fig = plt.figure()
    plt.title(r'Testing Error vs. $\lambda$', fontsize = 22)    
    plt.plot(lambdas, set1TestErrors, lambdas1, set2TestErrors, marker = '.')
    plt.legend(('Training Set 1', 'Training Set 2'), loc = 'best', fontsize = 14)
    plt.xscale('log')
    plt.xlabel('$\lambda$ (log scale)', fontsize = 18)
    plt.ylabel('Testing Error', fontsize = 18)
    plt.margins(y=0.02) 

    minL = min(set2TestErrors)
    for i in range(len(lambdas1)):
        if set2TestErrors[i] == minL:
            print ("Lambda corresponding to lowest test error from set 2" + str(lambdas1[i]))
    
    for j in range(len(set1TestErrors)):
        if set1TestErrors[j] == min(set1TestErrors):
            print ("Lambda corresponding to lowest test error from set 1" + str(lambdas[j]))
    

    
    #For lambdas
    set1Norms = []
    set2Norms = []
    for i in weights:
        set1Norms.append(returnL2Norm(i))
    for j in weights1:
        set2Norms.append(returnL2Norm(j))
    fig = plt.figure()
    plt.title(r'$\ell_2$ norm of w vs. $\lambda$', fontsize = 22)    
    plt.plot(lambdas, set1Norms, lambdas1, set2Norms, marker = '.')
    plt.legend(('Training Set 1', 'Training Set 2'), loc = 'best', fontsize = 14)
    plt.xscale('log')
    plt.xlabel('$\lambda$ (log scale)', fontsize = 18)
    plt.ylabel('$\ell_2$ norm', fontsize = 18)
    plt.margins(y=0.02)      
    
main()