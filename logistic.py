# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:04:15 2017

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('D:/Anaconda/test/机器学习/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inx):
    return 1/(1+np.exp(-inx))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)#100*3的矩阵
    labelMat = np.mat(classLabels).T#100*1的矩阵
    m,n = dataMatrix.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))#3*1，定义初始回归系数
    for k in range(maxCycles):#迭代调整回归系数
        h = sigmoid(dataMatrix*weights)#100*1的矩阵，利用回归系数计算样本预测值
        error = (labelMat - h)#100*1的矩阵，计算每个样本的误差(实际值-预测值)
        weights = weights + alpha*dataMatrix.T*error#3*1的矩阵，调整回归系数weights
        #dataMatrix.T是3*100的矩阵，error是100*1的矩阵，双方经过300次乘法相加，得到3*1的矩阵
    return weights#输出最终的回归系数

def plotBsetFit(wei):
    weights = wei.getA()#将矩阵转化为np.ndarry对象
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3,3,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    #这里，令sigmoid函数等于0，求出x1和x2的关系，即为要画的分割线
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
#以上梯度算法中，每次迭代都要遍历所有的数据，每次都有300次的乘法，如果数据量大，很容易
#累死，下面优化为随机梯度算法，即每次迭代只选择一个样本
    
def stocGradAscent0(dataMatrix,classLabels):
    m,n = dataMatrix.shape
    alpha = 0.01
    weights = np.ones((n,1))
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha*dataMatrix[i]*error
    return weights #注意，这里返回的weights是1*3的np.ndarry形式，不是矩阵形式
    #因此在调用画图函数时，需进行转换和转置
#调用画图函数可以看出，迭代效果没有全数据集迭代好，下面尝试对步长进行优化

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = dataMatrix.shape
    weights = np.ones((n,1))
    for j in range(numIter):#重复迭代全集样本
        dataIndex = list(range(m))
        for i in range(m):#随机迭代每一个样本
            alpha = 4/(1+j+i)+0.01 #s随着迭代调整步长
            randIndex = int(np.random.uniform(0,len(dataIndex)))#随机选择一个样本
            h = sigmoid(np.sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights +alpha*dataMatrix[randIndex]*error
            del dataIndex[randIndex] 
    return weights
#画图，可以看出效果要比上个随机梯度好很多
    
'''从疝气病症预测病马的死亡率'''
def classifyVector(x,weights):#定义一个分类函数，利用sigmoid计算属于1的概率
    prob = sigmoid(np.sum(x*weights))
    if prob > 0.5:
        return 1
    else:
        return 0
    
def colicTest():
    frTrain = open('D:/Anaconda/test/机器学习/Ch05/horseColicTraining.txt')
    frTest = open('D:/Anaconda/test/机器学习/Ch05/horseColicTest.txt')
    trainingSet = []; trainlabels = []
    for i in frTrain.readlines():#加载训练数据
        ilist = i.strip().split('\t')
        trainingSet.append([float(x) for x in ilist[:-1]])
        trainlabels.append(float(ilist[-1]))  
    frTrain.close()
    
    trainWeights = stocGradAscent1(np.array(trainingSet),trainlabels,1000)
    #利用训练数据生成一组最优回归系数
    
    error = 0; numTest = 0
    for i in frTest.readlines():
        numTest += 1 
        ilist = i.strip().split('\t')
        pard = classifyVector(np.array([float(x) for x in ilist[:-1]]),trainWeights)
        #利用测试数据和最优回归系数，判定分类
        if int(pard) != int(ilist[-1]):
            error += 1
    frTest.close()
    errorRate = float(error)/numTest
    print('the error rate of this test is:%f'%errorRate)
    return errorRate

def multiTest():
    numTest = 10; errorSum = 0
    for i in range(numTest): #重复10次，求平均错误率
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f'%(numTest,errorSum/numTest))

#RuntimeWarning: overflow encountered in exp python
    
    
'''使用sklearn库实现逻辑回归'''
from sklearn.linear_model import LogisticRegression
def colicSklearn():
    frTrain = open('D:/Anaconda/test/机器学习/Ch05/horseColicTraining.txt')                                        #打开训练集
    frTest = open('D:/Anaconda/test/机器学习/Ch05/horseColicTest.txt')                                                #打开测试集
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    #classifier = LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet, trainingLabels)
    classifier = LogisticRegression(solver='sag',max_iter=5000).fit(trainingSet, trainingLabels)
    #solver优化算法选择，默认是liblinear，适用于小数据集，且多元分类精确度底
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)
 
if __name__ == '__main__':
    colicSklearn()