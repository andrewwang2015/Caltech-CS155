import csv
import numpy as np
import matplotlib.pyplot as plt

def loadData(fileName):
    '''
    This function takes a file name string and returns the data as numpy array
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
    for i in arr:
        inputs.append(i[0])    #Because array has two dimensions
        outputs.append(i[1])
    return np.asarray(inputs), np.asarray(outputs)

def returnKFoldData(foldNumber, numberFolds, data):
    '''
    This function returns training and validation data as numpy arrays for 
    a single fold/partition. Fold number designates which partition to get the
    data for. For example, for a data set of 100 and 5 folds, a fold number of 1
    will have data points 1-20 be validation data and the rest be training data.
    Fold number 2 will have datapoints 21-40 be validation data and the rest be
    training data, etc.
    '''
    
    foldNumber = int(foldNumber)
    numberFolds = int(numberFolds)
    perFold = int(len(data) / numberFolds)
    lowerBound = (foldNumber - 1) * perFold
    upperBound = foldNumber * perFold
    validationData = data[lowerBound : upperBound] #lower and upper bound correspond to the datapoints that are used for validation
    trainingData = np.concatenate((data[:lowerBound], data[upperBound:]), axis = 0)   #make training data all data not from lower bound to upperbound
    return trainingData, validationData

def computeAverageErrors(data, numberFolds, degreePoly):
    '''
    This function takes in the data, number of folds, and the degree polynomial
    and returns average training and validation errors
    '''
    
    trainingErrors = []
    validationErrors = []
    for k in range(1, numberFolds + 1):
        trainingError = 0
        validationError = 0
        training, validation = returnKFoldData(k, numberFolds, data)  #Get training and validation data from the returnKFoldData function
        assert(len(validation) == len(data) / numberFolds)
        assert(len(training) == len(data) - len(validation))
        trainingX, trainingY = getInputsAndOutputs(training)
        validationX, validationY = getInputsAndOutputs(validation)
        z = np.polyfit(trainingX, trainingY, degreePoly)    #np.polyfit will return coefficients
        
        #Iterate through training data and calculate training error
        for i in range(len(trainingX)):
            prediction = np.polyval(z, trainingX[i])
            trainingError += squaredError(prediction, trainingY[i])
        
        #Iterate through validation data and calculate validation error
        for j in range(len(validationX)):
            prediction1 = np.polyval(z, validationX[j])
            validationError += squaredError(prediction1, validationY[j])
        
        trainingErrors.append(trainingError / len(trainingX))
        validationErrors.append(validationError / len(validationX))   
    return sum(trainingErrors) / numberFolds, sum(validationErrors) / numberFolds
            
def squaredError(num1, num2):
    '''
    Basic function that returns squared error of two inputs
    '''
    
    return (num1 - num2) ** 2


allData = loadData("bv_data.csv")

trainingErr = []
validationErr = []

degrees = [1, 2, 6, 12]
for d in degrees:
    for size in range(20,101,5):
        tempData = allData[:size]
        avgTrain, avgValidation = computeAverageErrors(tempData, 5, d)
        trainingErr.append(avgTrain)
        validationErr.append(avgValidation)

degree1Training = trainingErr[:17]
degree1Validation = validationErr[:17]

degree2Training = trainingErr[17:34]
degree2Validation = validationErr[17:34]

degree6Training = trainingErr[34: 51]
degree6Validation = validationErr[34:51]

degree12Training = trainingErr[51:]
degree12Validation = validationErr[51:]


# Time to plot the curves
N = list(range(20,101,5))

x = N

plt.figure(1)
plt.subplot(221)

plt.title('1st-degree Polynomial Regression')
plt.plot(x, degree1Training, x, degree1Validation, marker = '.')
plt.legend(('Training: Degree 1', 'Validation: Degree 1'), loc = 'upper right')
plt.xlabel('Number of Data Points')
plt.ylabel('Average Error')
plt.margins(y=0.02)

plt.subplot(222)
plt.title('2nd-degree Polynomial Regression')
plt.plot(x, degree2Training, x, degree2Validation, marker = '.')
plt.legend(('Training: Degree 2', 'Validation: Degree 2'), loc = 'upper right')
plt.xlabel('Number of Data Points')
plt.ylabel('Average Error')
plt.margins(y=0.02)

plt.subplot(223)
plt.title('6th-degree Polynomial Regression')
plt.plot(x, degree6Training, x, degree6Validation, marker = '.')
plt.legend(('Training: Degree 6', 'Validation: Degree 6'), loc = 'upper right')
plt.xlabel('Number of Data Points')
plt.ylabel('Average Error')
plt.margins(y=0.02)

plt.subplot(224)
plt.title('12th-degree Polynomial Regression (Log Scale for Readability)')
plt.plot(x, degree12Training, x, degree12Validation, marker = '.')
plt.legend(('Training: Degree 12', 'Validation: Degree 12'), loc = 'upper right')
plt.xlabel('Number of Data Points')
plt.ylabel('Average Error (log scale)')
plt.yscale('log')
plt.margins(y=0.02)