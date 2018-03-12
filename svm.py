# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:03:40 2017

@author: Administrator
"""
import numpy as np

file = 'D:/Anaconda/test/机器学习/Ch06/testSet.txt'
def loadDataSet(file):
    '''读取文件函数
    Parameters：file - 文件路径
    Returns：dataMat - 数据集
             labelMat - 数据标签
    Author：Li Wei
    '''
    dataMat = []; labelMat = []
    f = open(file)
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    '''随机选择alpha函数
    Parameters：i - alpha
                m - alpha参数个数
    Returns：j
    Author：Li Wei
    '''
    j = i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    '''修建alpha
    Parameters：aj - alpha值
                H - alpha上限
                L - alpha下限
    Returns：aj - alpha值
    Author：Li Wei
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

dataArr,labelArr = loadDataSet(file)

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    '''简化版SMO算法
    Parameters：dataMatIn - 数据矩阵
                classLabels - 数据标签
                C - 松弛变量
                toler - 容错率
                maxIter - 最大迭代次数
    Returns：无
    Author：Li Wei
    '''
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).T
    b = 0; m,n = dataMatrix.shape
    alphas = np.mat(np.zeros((m,1)))
    iters = 0
    while (iters<maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0 :
                    print('eta>=0')
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j]-alphaJold)<0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*\
                dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*\
                dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b = b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2
                    alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iters,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): 
            iters += 1
        else: 
            iters = 0
        print("iteration number: %d" % iters)
    return b,alphas


'''使用sklearn实现SVM'''
from sklearn.svm import SVC
from os import listdir

def img2vector(filename):
    """将32x32的二进制图像转换为1x1024向量。
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
    Author：Li Wei
    """
    #创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect

def handwritingClassTest():
    """手写数字分类测试
    Parameters:
        无
    Returns:
        无
    Author：Li Wei
    """
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('机器学习实战/Ch02/digits/trainingDigits/')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('机器学习实战/Ch02/digits/trainingDigits/%s' % (fileNameStr))
    clf = SVC(C=200,kernel='rbf')
    clf.fit(trainingMat,hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('机器学习实战/Ch02/digits/testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('机器学习实战/Ch02/digits/testDigits/%s' % (fileNameStr))
        #获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = clf.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
    handwritingClassTest()