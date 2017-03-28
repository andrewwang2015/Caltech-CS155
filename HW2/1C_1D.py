import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, svm
import math
import random
from plot_decision_boundary import make_plot

def loadData(fileName):
    '''
    Takes filename string and returns raw data as numpy array
    '''
    
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    return np.asarray(data, dtype = float)


def getInputsAndOutputs(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y)
    '''    
    inputs = []
    outputs = []
    arr1 = np.asarray([1.0])
    for i in arr:
        inputs.append(np.concatenate((arr1, i[:-1]), axis = 0))   # the x vector
        outputs.append(i[(len(i) - 1)])      # last element, y
    return np.asarray(inputs), np.asarray(outputs)

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

def runLogisticRegression(inputs, outputs):
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array
    
    '''
    clf = linear_model.LogisticRegression()
    clf = clf.fit(inputs, outputs)
    predicted = clf.predict(inputs)
    return clf, predicted

def runLogisticRegressionWeighted(inputs, outputs):
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array. This 
    weighs positive-labeled 5x more than negative-labeled instances.
    
    '''
    clf = linear_model.LogisticRegression(class_weight = {1.0: 5, -1.0: 1})
    clf = clf.fit(inputs, outputs)
    predicted = clf.predict(inputs)
    return clf, predicted

def runSVM(inputs, outputs):
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array. 
    
    '''
    clf = svm.SVC(kernel = 'linear')
    clf.fit(inputs, outputs)
    predicted = clf.predict(inputs)
    return clf, predicted

def runSVMWeighted(inputs, outputs):
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array. This 
    weighs positive-labeled 5x more than negative-labeled instances.
    
    '''
    clf = svm.SVC(kernel = 'linear', class_weight = {1.0: 5, -1.0:1})
    clf.fit(inputs, outputs)
    predicted = clf.predict(inputs)
    return clf, predicted

    
def main():
    dataSet1Inputs, dataSet1Outputs = getInputsAndOutputsV2(loadData("problem1dataset1.txt"))
    dataSet2Inputs, dataSet2Outputs = getInputsAndOutputsV2(loadData("problem1dataset2.txt"))
    dataSet3Inputs, dataSet3Outputs = getInputsAndOutputsV2(loadData("problem1dataset3.txt"))
    
    clf1_log, predicted1_log = runLogisticRegression(dataSet1Inputs, dataSet1Outputs)
    make_plot(dataSet1Inputs, dataSet1Outputs, clf1_log, "Dataset 1 - Logistic Regression", "dataset1_log")
    
    clf2_log, predicted2_log = runLogisticRegression(dataSet2Inputs, dataSet2Outputs)
    make_plot(dataSet2Inputs, dataSet2Outputs, clf2_log, "Dataset 2 - Logistic Regression", "dataset2_log")
    
    clf3_log, predicted3_log = runLogisticRegression(dataSet3Inputs, dataSet3Outputs)
    make_plot(dataSet3Inputs, dataSet3Outputs, clf3_log, "Dataset 3 - Logistic Regression", "dataset3_log")
    
    
    clf1_SVM, predicted1_SVM = runSVM(dataSet1Inputs, dataSet1Outputs)
    make_plot(dataSet1Inputs, dataSet1Outputs, clf1_SVM, "Dataset 1 - SVM Regression", "dataset1_SVM")
    
    clf2_SVM, predicted2_SVM = runSVM(dataSet2Inputs, dataSet2Outputs)
    make_plot(dataSet2Inputs, dataSet2Outputs, clf2_SVM, "Dataset 2 - SVM Regression", "dataset2_SVM")
    
    clf3_SVM, predicted3_SVM = runSVM(dataSet3Inputs, dataSet3Outputs)
    make_plot(dataSet3Inputs, dataSet3Outputs, clf3_SVM, "Dataset 3 - SVM Regression", "dataset3_SVM")
    
    clf2_SVM_weighted, predicted2_SVM_weighted = runSVMWeighted(dataSet2Inputs, dataSet2Outputs)
    make_plot(dataSet2Inputs, dataSet2Outputs, clf2_SVM_weighted, "Dataset 2 - SVM Regression (Weighted)", "dataset2_SVM_weighted")
    
    clf2_log_weighted, predicted2_log_weighted = runLogisticRegressionWeighted(dataSet2Inputs, dataSet2Outputs)
    make_plot(dataSet2Inputs, dataSet2Outputs, clf2_log_weighted, "Dataset 2 - Logistic Regression (Weighted)", "dataset2_log_weighted")    
    
    
    
main()