import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt

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
        
    
def calculateGradientU(U, Y, V, lam, i, j):   
    term1 = lam * U[:,i]
    v_j = V[:,j]
    uT_v = np.dot(U[:,i], v_j)
    term2 = v_j * (Y[i][j] - uT_v)
    return term1 - term2



def calculateGradientV(U, Y, V, lam, i, j):
    term1 = lam * V[:,j]
    v_j = V[:,j]
    uT_v = np.dot(U[:,i], v_j)
    term2 = U[:,i] * (Y[i][j] - uT_v)
    return term1 - term2
    
def runSGD(Y, U, V, lam, rate, i, j):
    
    randomPointIndex = random.randint(0, len(i) - 1)
    colU = i[randomPointIndex] - 1
    colV = j[randomPointIndex] - 1

    U[:, colU] -= rate * calculateGradientU(U, Y, V, lam, colU, colV)

    V[:, colV] -= rate * calculateGradientV(U, Y, V, lam, colU, colV)  
    return



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
    k = [10,20,30,50,100]
    i, j, y_ij = loadData("train.txt")
    i_test, j_test, y_ij_test = loadData("test.txt")
    M = int(max(i)) 
    N = int(max(j)) 
    numTraining = len(i)
    numTest = len(i_test)
    Y = [[0 for x in range(N)] for y in range(M)]
    Y = np.asarray(Y, dtype = float)
    
    Y_test = [[0 for x in range(N)] for y in range(M)]
    Y_test = np.asarray(Y_test, dtype = float)  
    
    for x in range(len(i)):
        Y[int(i[x])-1][int(j[x])-1] = y_ij[x]
    for x in range(len(i_test)):
        if i_test[x] <= M and j_test[x] <= N:
            Y_test[int(i_test[x])-1][int(j_test[x])-1] = y_ij_test[x]
  
  
    ## WITHOUT REGULARIZATION ##
    E_in = []
    E_out = []
    for latent in k:
        iteration = 0
        U, V = returnU_V(M, N, latent)
        allLoss = []
        allLoss.append(calculateError(U, V, Y))

        while (True):
            iteration += 1
            runSGD(Y, U, V, 0, learningRate, i, j)
            if iteration % 50000 == 0:
                currentError = calculateError(U, V, Y)
                allLoss.append(currentError)
                if (math.fabs(allLoss[-1] - allLoss[-2]) / math.fabs (allLoss[1] - allLoss[0]) <= 0.001):  #Stopping condition
                    E_in.append(currentError/numTraining)
                    E_out.append(calculateError(U, V, Y_test)/numTest)
                    break                

    x = k
    fig = plt.figure()
    plt.title("Without regularization", fontsize = 22)
    plt.plot(x, E_in, x, E_out)

    plt.legend(('Training Error', 'Test Error'), loc = 'best', fontsize = 14)
    plt.xlabel('K (number of latent factors)', fontsize = 18)
    plt.ylabel('Normalized Mean Squared Loss', fontsize = 18)
    plt.margins(y=0.02)   
    
    # WITH REGULARIZATION ##
    lam = [10**-4, 10**-3, 10**-2, 10**-1, 1]
    E_in = []
    E_out = []
    for index in range(len(k)):
        iteration = 0
        U, V = returnU_V(M, N, k[index])
        allLoss = []
        allLoss.append(calculateError(U, V, Y))

        while (True):
            iteration += 1
            runSGD(Y, U, V, lam[index], learningRate, i, j)
            if iteration % 50000 == 0:
                currentError = calculateError(U, V, Y)
                allLoss.append(currentError)
                if (math.fabs(allLoss[-1] - allLoss[-2]) / math.fabs (allLoss[1] - allLoss[0]) <= 0.001):  #Stopping condition
                    E_in.append(currentError/numTraining)
                    E_out.append(calculateError(U, V, Y_test)/numTest)
                    break                
    

    x = k
    fig = plt.figure()
    plt.title("With Regularization", fontsize = 22)
    plt.plot(x, E_in, x, E_out)

    plt.legend(('Training Error', 'Test Error'), loc = 'best', fontsize = 14)
    plt.xlabel('K (number of latent factors)', fontsize = 18)
    plt.ylabel('Normalized Mean Squared Loss', fontsize = 18)
    plt.margins(y=0.02)  
    
    lam = [10**-4, 10**-3, 10**-2, 10**-1, 1]
    allErrors = []
    for lamVal in lam:
        for latent in k:
            iteration = 0
            U, V = returnU_V(M, N, latent)
            allLoss = []
            allLoss.append(calculateError(U, V, Y))
        
            while (True):
                iteration += 1
                runSGD(Y, U, V, lamVal, learningRate, i, j)
                if iteration % 50000 == 0:
                    currentError = calculateError(U, V, Y)
                    allLoss.append(currentError)
                    if (math.fabs(allLoss[-1] - allLoss[-2]) / math.fabs (allLoss[1] - allLoss[0]) <= 0.001):  #Stopping condition
                        E_in = (currentError/numTraining)
                        E_out = (calculateError(U, V, Y_test)/numTest)
                        print(latent, lamVal, E_in, E_out)
                        allErrors.append([latent, lamVal, E_in, E_out])
                        break               

    
    E_in_10 = []
    E_in_20 = []
    E_in_30 = []
    E_in_50 = []
    E_in_100 = []
    
    E_out_10 = []
    E_out_20 = []
    E_out_30 = []
    E_out_40 = []
    E_out_50 = []
    E_out_100 = []
    
    for i in allErrors:
        if i[0] == 10:
            E_in_10.append(i[2])
            E_out_10.append(i[3])
        elif i[0] == 20:
            E_in_20.append(i[2])
            E_out_20.append(i[3])            
        elif i[0] == 30:
            E_in_30.append(i[2])
            E_out_30.append(i[3])            
        elif i[0] == 50:
            E_in_50.append(i[2])
            E_out_50.append(i[3])            
        elif i[0] == 100:
            E_in_100.append(i[2])
            E_out_100.append(i[3])        
    x = lam
    fig = plt.figure()
    plt.title(r"Loss vs. $\lambda$ for k = 10", fontsize = 22)
    plt.plot(x, E_in_10, x, E_out_10)

    plt.legend(('Training Error', 'Test Error'), loc = 'best', fontsize = 14)
    plt.xlabel(r'$\lambda$', fontsize = 18)
    plt.ylabel('Normalized Mean Squared Loss', fontsize = 18)
    plt.margins(y=0.02)   
    
    
    E_in_lam1 = []
    E_in_lam2 = []
    E_in_lam3 = []
    E_in_lam4 = []
    E_in_lam5 = []
    
    E_out_lam1 = []
    E_out_lam2 = []
    E_out_lam3 = []
    E_out_lam4 = []
    E_out_lam5 = []

        
    for i in allErrors:
        if i[1] == 10**-4:
            E_in_lam1.append(i[2])
            E_out_lam1.append(i[3])
        elif i[1] == 10**-3:
            E_in_lam2.append(i[2])
            E_out_lam2.append(i[3])            
        elif i[1] == 10**-2:
            E_in_lam3.append(i[2])
            E_out_lam3.append(i[3])            
        elif i[1] == 10**-1:
            E_in_lam4.append(i[2])
            E_out_lam4.append(i[3])            
        elif i[1] == 1:
            E_in_lam5.append(i[2])
            E_out_lam5.append(i[3])     
        
        
        
    x = k
    fig = plt.figure()
    plt.title(r"Loss vs. k with varying $\lambda$", fontsize = 22)
    plt.plot(x, E_in_lam1, x, E_out_lam1, x, E_in_lam2, x, E_out_lam2, 
             x, E_in_lam3, x, E_out_lam3, x, E_in_lam4, x, E_out_lam4, x, E_in_lam5, x, E_out_lam5)

    plt.legend((r'$E_{in}(\lambda$ = 0.0001)', r'$E_{out}(\lambda$ = 0.0001)', 
               r'$E_{in}(\lambda$ = 0.001)', r'$E_{out}(\lambda$ = 0.001)',
               r'$E_{in}(\lambda$ = 0.01)', r'$E_{out}(\lambda$ = 0.01)', 
               r'$E_{in}(\lambda$ = 0.1)', r'$E_{out}(\lambda$ = 0.1)', 
               r'$E_{in}(\lambda$ = 1)', r'$E_{out}(\lambda$ = 1)'), loc = 'best', fontsize = 14)
    plt.xlabel(r'$k$', fontsize = 18)
    plt.ylabel('Normalized Mean Squared Loss', fontsize = 18)
    plt.margins(y=0.02)     
    
    x = k

    finalGraph_in = [E_in_10[0], E_in_20[1], E_in_30[2], 
                      E_in_50[3], E_in_100[4]]
    finalGraph_out = [E_out_10[0], E_out_20[1], E_out_30[2], E_out_50[3], E_out_100[4]]
    fig = plt.figure()
    plt.title(r"Loss vs. $k$ with regularization", fontsize = 22)
    plt.plot(x, finalGraph_in, x, finalGraph_out)

    plt.legend((r'$E_{in}$', r'$E_{out}$'), loc = 'best', fontsize = 14)
    plt.xlabel(r'$k$', fontsize = 18)
    plt.ylabel('Normalized Mean Squared Loss', fontsize = 18)
    plt.margins(y=0.02)      
    
    
        
            


main()
        
        
    