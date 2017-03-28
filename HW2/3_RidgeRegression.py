from sklearn import linear_model
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
        data_iter = csv.reader(dest_f, delimiter = '\t', quotechar = '"')
        data = [data for data in data_iter]
    return np.asarray(data, dtype = float)

def getInputsAndOutputsV2(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y) without x_0 being 1
    '''    
    inputs = []
    outputs = []

    for i in arr:
        inputs.append(i[:-1])   # the x vector
        outputs.append(i[(len(i) - 1)])      # last element, y
    return np.asarray(inputs), np.asarray(outputs)

def runLassoRegression(alphaValue, inputs, outputs):
    '''
    Function runs the Lasso Regression and returns weight vector
    '''
    clf = linear_model.Lasso(alpha=alphaValue)
    clf.fit(inputs, outputs)
    return clf.coef_

def runRidgeRegression(alphaValue, inputs, outputs):
    '''
    Function runs the Ridge regression and returns weight vector 
    '''
    clf = linear_model.Ridge(alpha=alphaValue)
    clf.fit(inputs, outputs)
    return clf.coef_

def main():
    coefficients = []
    coefficients1 = []
    coefficients2 = []
    coefficients3 = []
    coefficients4 = []
    coefficients5 = []
    coefficients6 = []
    coefficients7 = []
    coefficients8 = []
    coefficients9 = []
    alphas = []
    allData = loadData("problem3data.txt")
    inputs,outputs = getInputsAndOutputsV2(allData)
    count = 0
    alpha = 0.0
    while (True):
        numSmall = 0
        count += 1
        #print (alpha)
        coefficients.append(runRidgeRegression(alpha, inputs, outputs))
        alphas.append(alpha)
        alpha += 30  #0.01 for lasso, 30 for ridge
        #print (coefficients[-1])
        assert(len(coefficients[-1]) == 9)
        for i in coefficients[-1]:
            if math.fabs(i) < 0.3:
                numSmall += 1
        if numSmall == 6:
            break

    for i in coefficients:
        coefficients1.append(i[0])
        coefficients2.append(i[1])
        coefficients3.append(i[2])
        coefficients4.append(i[3])
        coefficients5.append(i[4])
        coefficients6.append(i[5])
        coefficients7.append(i[6])
        coefficients8.append(i[7])
        coefficients9.append(i[8])
    
    x = alphas
    fig = plt.figure()
    plt.title(r'Model weights vs. $\alpha$ (Ridge)', fontsize = 22)
    #plt.axis((0, alphas[-1], -5 ,5))  #for ridge
    plt.plot(x, coefficients1, x, coefficients2, x, coefficients3, x , coefficients4, 
             x, coefficients5, x, coefficients6, x, coefficients7, x, coefficients8, x,
             coefficients9)

    plt.legend(('w$_1$','w$_2$', 'w$_3$', 'w$_4$', 'w$_5$', 'w$_6$', 'w$_7$', 'w$_8$', 'w$_9$'), loc = 'best', fontsize = 14)
    plt.xlabel(r'$\alpha$', fontsize = 18)
    plt.ylabel('Weights', fontsize = 18)
    plt.margins(y=0.02)   
    
main()
              
        