'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataLine = [1.0]
        for i in lineArr:
            dataLine.append(float(i))
        label = dataLine.pop() # pop the last column referring to  label
        dataMat.append(dataLine)
        labelMat.append(int(label))
    return mat(dataMat), mat(labelMat).transpose()
    
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
    
def batchGradAscent(dataMat, labelMat):
    m,n = shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMat * weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMat.transpose() * error #matrix mult
    return weights

def stocGradAscent1(dataMat, labelMat):
    m,n = shape(dataMat)
    alpha = 0.01
    weights = ones((n,1))   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMat[i]*weights))
        error = labelMat[i] - h
        weights = weights + (alpha * error * dataMat[i]).transpose()
    return weights

def stocGradAscent2(dataMat, labelMat, numIter=2): #
    m,n = shape(dataMat)
    weights = ones((n,1))   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = labelMat[randIndex] - h
            weights = weights + (alpha * error * dataMat[randIndex]).transpose()
            del(dataIndex[randIndex])
    return weights

def classify(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else: 
        return 0.0

def test():
    dataMat, labelMat = loadDataSet('testData.dat')
    weights0 = batchGradAscent(dataMat, labelMat)
    weights1 = stocGradAscent1(dataMat, labelMat)
    weights2 = stocGradAscent2(dataMat, labelMat)
    print('batchGradAscent:', weights0)
    print('stocGradAscent0:', weights1)
    print('stocGradAscent1:', weights2)

if __name__ == '__main__':
    test()

        