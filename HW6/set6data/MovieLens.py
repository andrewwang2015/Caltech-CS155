import numpy as np
import random
import time

def loadData(fileName):
    i = []
    j = []
    y_ij = []
    f = open(fileName, "r")
    
    for line in f:
        i1, j1, y_ij1 = line.split()
        i.append(i1)
        j.append(j1)
        y_ij.append(y_ij1)
        
    return np.asarray(i, dtype = float), np.asarray(j, dtype = float), np.asarray(y_ij, dtype = float)

def returnU_V(M, N, k):
    '''
    U: k x M
    V: k x N
    '''
    U = [[random.uniform(-0.5, 0.5) for x in range(M)] for y in range(k)]
    V = [[random.uniform(-0.5, 0.5) for x in range(N)] for y in range(k)]
    return np.asarray(U, dtype = float), np.asarray(V, dtype = float)

def getSumAcrossColumns_U(Y, V, U, i):
    '''
    Used for gradient calculation of U_i 
    '''
    returnSum = [0 for x in range(V.shape[0])]
    returnSum = np.asarray(returnSum, dtype = float)
    for j in range(V.shape[1]):
        returnSum += V[:,j] * (Y[i][j] - np.dot(U[:,i], V[:,j]))
    return returnSum
        
def getSumAcrossColumns_V(Y, V, U, j):
    '''
    Used for gradient calculation of V_j
    '''
    returnSum = [0 for x in range(V.shape[0])]
    returnSum = np.asarray(returnSum, dtype = float)
    for i in range(U.shape[1]):
        returnSum += U[:,i] * (Y[i][j] - np.dot(U[:,i], V[:,j]))
    return returnSum
        
    
def calculateGradientU(U, Y, V, lam, i):   
    gradientU_i = lam * U[:, i] - getSumAcrossColumns_U(Y, V, U, i)	
    return gradientU_i


def calculateGradientV(U, Y, V, lam, j):
    gradientV_j = lam * V[:, j] - getSumAcrossColumns_V(Y, V, U, j)
    return gradientV_j
    
    
def runSGD(Y, U, V, lam, rate, i, j):
    
    randomPointIndex = random.randint(0, len(i) - 1)
    colU = i[randomPointIndex] - 1
    colV = j[randomPointIndex] - 1

    U[:, colU] -= rate * calculateGradientU(U, Y, V, lam, colU)

    V[:, colV] -= rate * calculateGradientV(U, Y, V, lam, colV)  
    return U, V

def returnSquaredError(num1, num2):
    return (num1-num2)**2

def calculateError(U, V, Y):
    assert(U.shape[0] == V.shape[0])
    totalSquaredError = 0
    U_transpose = np.matrix(np.transpose(U))
    resultant = np.asarray(U_transpose * np.matrix(V))
    assert(resultant.shape == Y.shape)
    
    for row in range(len(Y)):
        for col in range(len(Y[0])):
            if Y[row][col] > 0.1:
                totalSquaredError += pow(Y[row][col] - resultant[row][col],2)
    return totalSquaredError/2
            
            

def main():
    learningRate = 0.03
    
    i, j, y_ij = loadData("train.txt")
    M = int(max(i)) 
    N = int(max(j)) 
    U, V = returnU_V(M, N, 10)
    Y = [[0 for x in range(N)] for y in range(M)]
    Y = np.asarray(Y, dtype = float)
    print(Y.shape)
    
    for x in range(len(i)):
        Y[int(i[x])-1][int(j[x])-1] = y_ij[x]
    print(calculateError(U, V, Y))
    for x in range(700000):
        runSGD(Y, U, V, 0, learningRate, i, j)
        if x % 50000 == 0:
            print(calculateError(U, V, Y))

main()
        
        
    