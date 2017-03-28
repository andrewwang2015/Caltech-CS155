########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang, Avishek Dutta
# Description:  Set 5 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (i.e. run `python 2G.py`) to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!


import random
import operator

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            D:          Number of observations.
            A:          The transition matrix.
            O:          The observation matrix.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for i in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for i in range(self.L)] for j in range(M + 1)]
        seqs = [['' for i in range(self.L)] for j in range(M + 1)]
        temp = []
        # Initialize second row of probs and seqs
        
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            
        # Top-down approach starting with row 2
        for row in range(2, M+1):
            for col in range(self.L):
                temp = []
                for col2 in range(self.L):
                    temp.append(probs[row-1][col2] * self.O[col][x[row-1]] * 
                                self.A[col2] [col])
                max_index, max_value = max(enumerate(temp), key=operator.itemgetter(1))
                probs[row][col] = max_value
                seqs[row][col] = seqs[row-1][max_index] + str(max_index)
        
        max_index, max_value = max(enumerate(probs[M]), key=operator.itemgetter(1))
        max_seq = seqs[len(probs)-1][max_index] + str(max_index)
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        if (normalize):
            M = len(x)      # Length of sequence.
            alphas = [[0. for i in range(self.L)] for j in range(M + 1)]
            
            # Initialize the first row of alphas
            for i in range(self.L):
                alphas[1][i] = self.O[i][x[0]] * self.A_start[i]
            
            for row in range(2, M+1):
                for col in range(self.L):
                    temp = []
                    for col2 in range(self.L):
                        temp.append(alphas[row-1][col2] * self.O[col][x[row-1]] * self.A[col2] [col])
                    alphas[row][col] = sum(temp)
                normalizationConstant = sum(alphas[row])
                for normalCol in range(self.L):
                    alphas[row][normalCol] /= normalizationConstant
            return alphas            
            
        M = len(x)      # Length of sequence.
        alphas = [[0. for i in range(self.L)] for j in range(M + 1)]
        
        # Initialize the first row of alphas
        for i in range(self.L):
            alphas[1][i] = self.O[i][x[0]] * self.A_start[i]
        
        for row in range(2, M+1):
            for col in range(self.L):
                temp = []
                for col2 in range(self.L):
                    temp.append(alphas[row-1][col2] * self.O[col][x[row-1]] * 
                                self.A[col2] [col])
                alphas[row][col] = sum(temp)
        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
        '''
        
        if (normalize):
            M = len(x)      # Length of sequence.
            betas = [[0. for i in range(self.L)] for j in range(M + 1)]
    
            for i in range(self.L):
                betas[M][i] = 1
                      
            for row in range(M-1,0, -1):
                for col in range(self.L):
                    for col2 in range(self.L):
                        betas[row][col] += (betas[row+1][col2] * 
                                            self.O[col2][x[row]] * self.A[col] [col2])   
                normalizationConstant = sum(betas[row])
                for normalCol in range(self.L):
                    betas[row][normalCol] /= normalizationConstant                
            return betas            
            
        M = len(x)      # Length of sequence.
        betas = [[0. for i in range(self.L)] for j in range(M + 1)]

        for i in range(self.L):
            betas[M][i] = 1
                  
        for row in range(M-1,0, -1):
            for col in range(self.L):
                for col2 in range(self.L):
                    betas[row][col] += (betas[row+1][col2] * self.O[col2][x[row]] * 
                                self.A[col] [col2])   
        return betas
    
    def count_num_transitions(self, a, b, X, Y):
        '''
        Helper function for supervised_learning that counts number of cases
        where y_i^j = b and y_i^(j-1) = a
        '''
        num = 0
        den = 0
        for i in range(len(X)):
            for j in range(1, len(X[i])):
                if Y[i][j-1] == a:
                    den += 1
                    if Y[i][j] == b:
                        num += 1
                
        return num, den
    
    def count_num_observations(self, w, a, X, Y):
        '''
        Helper function for supervised_learning that counts number of cases
        where y_i^j = and x_i^j = w
        '''
        num = 0
        den = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                if Y[i][j] == a:
                    den += 1
                    if X[i][j] == w:
                        num += 1
        return num, den
        
    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.

        for a in range(self.L):
            for b in range(self.L):
                num, den = self.count_num_transitions(a, b, X, Y)
                self.A[a][b] = num / den
        

        # Calculate each element of O using the M-step formulas.
        
        for a in range(len(self.O)):
            for w in range(len(self.O[0])):
                num, den = self.count_num_observations(w, a, X, Y)
                self.O[a][w] = num / den  
            
        return 
    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''
        for iteration in range(1000):
            for sequence in X:
                M = len(sequence)
                gammas = [[0. for i in range(self.L)] for j in range(M + 1)]
                xi = [[[0. for i in range(self.L)] for j in range (self.L)] for k in range(M+1)]
                
                alphas = self.forward(sequence)
                betas = self.backward(sequence)
                
                
                # Calculate gammas
                for i in range(1, M + 1):
                    divisor = sum(alphas[i][k] * betas[i][k] for k in range(self.L))
                    for j in range(self.L):
                        gammas[i][j] = alphas[i][j] * betas[i][j] / divisor
                
                
                # Calculate xi
                
                for i in range(1, M):
                    for j in range(self.L):
                        for k in range(self.L):
                            num = alphas[i][j] * self.A[j][k] * betas[i+1][k] * self.O[k][sequence[i]]
                        
                            den = 0
                            for a in range(self.L):
                                for b in range(self.L):
                                    den += alphas[i][a] * self.A[a][b] * betas[i+1][b] * self.O[b][sequence[i]]
                            xi[i][j][k] = num / den
                
                for i in range(self.L):
                    for j in range(self.L):
                        num = sum(xi[k][i][j] for k in range(M-1))
                        den = sum(gammas[k][i] for k in range(M-1))
                        self.A[i][j] = num / den
                
                for i in range(self.L):
                    for j in range(self.D):
                        num = sum(gammas[k][i] for k in range(M) if sequence[k] == sequence[j])
                        den = sum(gammas[k][i] for k in range(M))
                        self.O[i][j] = num / den
                
    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''
        emission = ''
        state = 1
        for i in range(M):
            rowToSearch = 0
            tempLst = []
            runningSum = 0
            curState = random.random()
            for j in self.A[state]:
                runningSum += j
                tempLst.append(runningSum)
            for i in range(len(tempLst)):
                if curState < tempLst[i]:
                    rowToSearch = i
                    state = i
                    break
            curObs = random.random()
            tempLst2 = []
            runningSum2 = 0
            for j in self.O[rowToSearch]:
                runningSum2 += j
                tempLst2.append(runningSum2)
            for i in range(len(tempLst2)):
                if curObs < tempLst2[i]:
                    emission += str(i)
                    break
        assert(len(emission) == M)
        return emission

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum([betas[1][k] * self.A_start[k] * self.O[k][x[0]] \
            for k in range(self.L)])

        return prob

def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learing.

    Arguments:
        X:          A list of variable length emission sequences 
        Y:          A corresponding list of variable length state sequences
                    Note that the elements in X line up with those in Y
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X)

    return HMM
