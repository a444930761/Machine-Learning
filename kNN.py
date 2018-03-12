# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:35:12 2017

@author: Administrator
"""

import numpy as np
import operator #运算符模块
import matplotlib.pyplot as plt

def ceratDataSet(): #定义一个创建数据集函数，并定义前两个属于A类，后两个属于B类
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = list('AABB')
    return group,labels

def classify0(x,dataSet,labels,k):#定义kNN算法
    dataSetSize = dataSet.shape[0] #获取数据集的行数，即有多少个数
    diffMat = np.tile(x,(dataSetSize,1))-dataSet #将要判断类型的数据x，转化成和数据集
    #同样结构，并求出和数据集的差(想像坐标系中两个二维点的x、y之差)
    sqDiffMat = diffMat ** 2 #求x与数据集每个点差的平方(x差和y差的平方)
    sqDistances = sqDiffMat.sum(axis=1) #求出x与数据集每个点的差方和（x差的平方+y差的平方）
    distances = sqDistances ** 0.5 #求平方根（勾股定理），即为两点之间的距离
    sorteDistIndicies = distances.argsort() #返回distances中数值从小到大对应的索引值
    
    classCount = {}
    for i in range(k): #从前k个最相近的判断类型
        voteIlabel = labels[sorteDistIndicies[i]] #取出类型
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),
                               reverse=True)
    #classCount.items()将字典转为元组，并按照key指定的方式进行排序，reverse为True，则为
    #倒序，operator.itemgetter(1)是指按照元组索引为1的值进行排序
    return sortedClassCount[0][0] #取出排序后第一个元素的类型

def file2matrix(filename):#定义读取文件函数
    f = open(filename) 
    filelist = []
    for i in f.readlines():
        a,b,c,d = i.strip().split('\t')
        filelist.append([a]+[b]+[c]+[d]) #将读取到的信息以列表的形式存入列表
    f.close()
    filearray = np.array(filelist) #转化为ndarry格式
    returnMat = filearray[:,:-1] #构建出数据集
    classLabelVector = filearray[:,-1] #数据集对应的类型
    return returnMat,classLabelVector

def autoNorm(dataSet):
    ranges = dataSet.astype(float).max(axis=0)-dataSet.astype(float).min(axis=0)
    #这里需要留意，数据的类型必须是数值型的才能参与计算
    minVals = dataSet.astype(float).min(axis=0)
    normDataSet = (dataSet.astype(float)-minVals)/ranges
    return normDataSet,ranges,minVals
'''
def datingClassTest(file): #定义检验函数
    p = 0.1
    datingDataMat,datingLabels = file2matrix(file)
    norMat,ranges,minVals = autoNorm(datingDataMat)
    numTestVecs = int(len(norMat)*p) #取10%的作为测试集
    errorCount = 0
    for i in range(numTestVecs):
        result = classify0(norMat[i,:],norMat[numTestVecs:,:],
                           datingLabels[numTestVecs:],3) #norMat从0到numTestVecs
        #的为测试数据，后面的为样本集
        print('the result is {},real answer is :{}'.format(result,datingLabels[i]))
        if result != datingLabels[i]:
            errorCount += 1
    print('the total error tate is {}'.format(errorCount/numTestVecs))
'''            

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses'] #类型列表
    ffMiles = float(input('输入每年获取的飞行常客里程数:'))
    percentTats = float(input('输入玩视频游戏所耗时间百分比:'))
    iceCream = float(input('输入每周消费的冰淇淋公升数:'))
    inArr = np.array([ffMiles,percentTats,iceCream]) #将输入的新数据转化为向量格式
    #注意顺序要和数据集中的顺序一致，即特征要对应上
    datingDataMat,datingLabels = file2matrix(file)
    norMat,ranges,minVals = autoNorm(datingDataMat)
    norinArr = (inArr-minVals)/ranges #将新数据进行归一化
    result = classify0(norinArr,norMat,datingLabels,3)
    print('该对象的类别是：{}'.format(resultList[result-1]))#-1是因为这里使用索引匹配
            
#group,labels = ceratDataSet()
#print(classify0([0,0],group,labels,3))
file = 'D:/Anaconda/test/机器学习/Ch02/datingTestSet2.txt'
datingClassTest(file)
#datingDataMat,datingLabels = file2matrix(file)    
#plt.scatter(datingDataMat[:,1],datingDataMat[:,2],
#            15*datingLabels.astype(int),15*datingLabels.astype(int)) 
#plt.xlabel('玩视频游戏所耗时间百分比')
#plt.ylabel('每周消耗的冰淇淋公升数')
#plt.legend()
#画散点图，后两个参数分别是形状尺寸和颜色，颜色可以用数字表示  

import os
file2 = 'D:/Anaconda/test/机器学习/Ch02/digits/testDigits/0_13.txt'
def img2vector(file): #读取文档，并转化为向量
    returnVect = np.zeros((1,1024))
    num = 0
    f = open(file)
    for i in f.readlines():
        ilen = int(len(i.strip()))
        returnVect[:,num:num+ilen] = list(i.strip())
        num += ilen
    f.close()
    return returnVect

filepath1 = 'D:/Anaconda/test/机器学习/Ch02/digits/trainingDigits/'
filepath2 = 'D:/Anaconda/test/机器学习/Ch02/digits/testDigits/'

def readfile(path):
    filelist = os.listdir(path)
    trainingMat = np.zeros((len(filelist),1024))
    hwLabels = []
    num = 0
    for i in filelist:
        Vect = img2vector(path+i)
        classNumstr = int(i.strip().split('_')[0])
        trainingMat[num,:] = Vect
        hwLabels.append(classNumstr)
        num += 1
    return trainingMat,hwLabels
        
def handwritingClassTest(file1,file2):
    trainingMat,hwLabels = readfile(filepath1)
    errorCount = 0
    for i in os.listdir(filepath2):
        vectorTest = img2vector(filepath2+i)
        classNumstr = int(i.strip().split('_')[0])
        result = classify0(vectorTest,trainingMat,hwLabels,3)
        print('the result is {},real answer is :{}'.format(result,classNumstr))
        if int(result) != classNumstr:
            errorCount += 1
    
    print('the total error tate is {}'.format(errorCount/len(os.listdir(filepath2))))      
    
    
    
    
handwritingClassTest(filepath1,filepath2)
