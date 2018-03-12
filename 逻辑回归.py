# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:32:49 2018

@author: a4449
"""
'''
Logistic回归的目的是寻找一个
非线性函数Sigmoid的最佳拟合参数，
求解过程可以由最优化算法（梯度下降/上升）完成。
'''
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

def loadfile(file):
    '''读取文件函数
    Parameters：file - 文件名
    Returns：dataSet - 数据集
             labels - 数据标签
    Author：Li Wei
    '''
    dataSet = []
    labels = []
    f = open(file)
    for i in f.readlines():
        data = i.strip().split('\t')
        a = data[:-1]
        a.insert(0,1) #这个语句会返回一个None，所以不能直接append它
        dataSet.append(a)
        labels.append(int(data[-1]))
    f.close()
    return np.array(dataSet).astype('float'),labels

def showdata(weights):
    '''可视化数据函数
    Parameters：weights - 权重数组
    Returns：无
    Author：Li Wei
    '''
    labelscolors = []
    for i in labels:
        if i == 1:
            labelscolors.append('red')
        else:
            labelscolors.append('blue')
    plt.scatter(dataSet[:,1],dataSet[:,2],s=15,c=labelscolors,alpha=0.8) #绘制原始数据
    x = np.arange(-3,3,0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x,y) #绘制回归线
    plt.show()

def sigmoid(inX):
    '''sigmoid函数
    Parameters：inX - 数据
    Returns：sigmoid函数
    Author: Li Wei
    '''
    return 1 / (1+np.exp(-inX))

def gradAscent(data,labels):
    '''梯度上升算法函数
    Parameters：data - 数据集
                labels - 标签
    Returns：权重数组（最优参数）
    Author：Li Wei
    '''
    dataMatrix = np.mat(data)
    labelMat = np.mat(labels).transpose()
    m,n = dataMatrix.shape
    alpha = 0.001
    maxCyles = 500
    #weights_array1 = np.array([])
    weights = np.ones((n,1))
    for k in range(maxCyles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        #weights_array1 = np.append(weights_array1,weights)
    #weights_array1 = weights_array1.reshape(maxCyles,n)
    return weights.getA()#,weights_array1

#原始的梯度算法计算量很大，下面采用随机梯度上升算法
def stocGradAscetn1(data,labels,numiter=150):
    '''随机梯度上升算法函数
    Parameters：data - 数据
                labels - 标签
                numiter - 计算次数
    Returns：权重数组
    Author：Li Wei
    '''
    m,n = np.shape(data)
    weights = np.ones(n)
    #weights_array2 = np.array([])
    for j in range(numiter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1+j+i) +0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(data[randIndex] * weights))
            error = labels[randIndex] - h   
            weights = weights + alpha * error * data[randIndex]
            #weights_array2 = np.append(weights_array2,weights,axis=0)
            del(dataIndex[randIndex])
    #weights_array2 = weights_array2.reshape(numiter*m,n)
    return weights#,weights_array2
    
#构建函数，查看回归系数和迭代次数之间的关系
def plotWeights(weights_array1,weights_array2):
    '''迭代次数和回归系数关系图
    Parameters：weights_array1 - 回归系数数组1
                weights_array2 - 回归系数数组2
    Returns：无
    Author：Li Wei
    '''
    font = FontProperties(fname='c:/windows/fonts/msyhl.ttc',size=14) #设置汉字字体
    
    fig,axs = plt.subplots(3,2,figsize=(20,10))
    
    #绘制上升梯度算法系数与迭代关系的函数
    x1 = np.arange(0,len(weights_array1),1)  
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title('梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel('W0',FontProperties=font)
    plt.setp(axs0_title_text,size=20,color='black',weight='bold')
    plt.setp(axs0_ylabel_text,size=20,color='black',weight='bold')
    
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel('W1',FontProperties=font)
    plt.setp(axs1_ylabel_text,size=20,color='black',weight='bold')
    
    #绘制W2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs1_ylabel_text = axs[2][0].set_ylabel('W2',FontProperties=font)
    axs1_xlabel_text = axs[2][0].set_xlabel('迭代次数',FontProperties=font)
    plt.setp(axs1_ylabel_text,size=20,color='black',weight='bold')
    plt.setp(axs1_xlabel_text,size=20,color='black',weight='bold')
    
    #绘制随机梯度算法系数与迭代次数的关系
    x2 = np.arange(0,len(weights_array2),1)
    #绘制w0系数与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs2_title_text = axs[0][1].set_title('随机梯度上升算法，回归系数与迭代次数关系',FontProperties=font)
    axs2_ylabel_text = axs[0][1].set_ylabel('W0',FontProperties=font)
    plt.setp(axs2_title_text,size=20,color='black',weight='bold')
    plt.setp(axs2_ylabel_text,size=20,color='black',weight='bold')
    
    #绘制W1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs2_ylabel_text = axs[1][1].set_ylabel('W1',FontProperties=font)
    plt.setp(axs2_ylabel_text,size=20,color='black',weight='bold')
    
    #绘制W2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_ylabel_text = axs[2][1].set_ylabel('W2',FontProperties=font)
    axs2_xlabel_text = axs[2][1].set_xlabel('迭代次数',FontProperties=font)
    plt.setp(axs2_ylabel_text,size=20,color='black',weight='bold')
    plt.setp(axs2_xlabel_text,size=20,color='black',weight='bold')
    
    plt.show()
    
if __name__ == '__main__':
    file = '机器学习实战/Ch05/testSet.txt'
    dataSet,labels = loadfile(file)
    #showdata(dataSet,labels)
    #weights = stocGradAscetn1(dataSet,labels)
    #showdata(weights)
    weights1,weights_array1 = gradAscent(dataSet,labels)
    weights2,weights_array2 = stocGradAscetn1(dataSet,labels)
    plotWeights(weights_array1,weights_array2)
    #以上可以看出，随机梯度上升算法，在迭代2000次，即将数据集遍历20遍以后达到最优
    #最初的梯度算法，则需将整个数据集遍历300遍以后才能达到最优
    
'''从疝气病症预测病马的死亡率'''
def classifyVector(x,weights):
    '''定义一个分类函数，利用sigmoid函数进行分类，大于0.5分类为1，否则分类为0
    Parameters：x - 要分类的数据
                weights - 权重数组
    Returns：类别
    Author：Li Wei
    '''
    prob = sigmoid(np.sum(x*weights))
    if prob > 0.5:
        return 1
    else:
        return 0
    
def colicTest():
    '''验证算法错误率函数
    Parameters：无
    Return：算法错误率
    '''
    frTrain = open('机器学习实战/Ch05/horseColicTraining.txt')
    frTest = open('机器学习实战/Ch05/horseColicTest.txt')
    trainingSet = []; trainlabels = []
    for i in frTrain.readlines():#加载训练数据
        ilist = i.strip().split('\t')
        trainingSet.append([float(x) for x in ilist[:-1]])
        trainlabels.append(float(ilist[-1]))  
    frTrain.close()
    
    trainWeights = stocGradAscetn1(np.array(trainingSet),trainlabels,1000)
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
    '''计算平均精度函数
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    numTest = 10; errorSum = 0
    for i in range(numTest): #重复10次，求平均错误率
        errorSum += colicTest()
    print('after %d iterations the average error rate is:%f'%(numTest,errorSum/numTest))

if __name__ == '__main__':
    multiTest()
    
'''使用sklearn库实现逻辑回归'''
from sklearn.linear_model import LogisticRegression

def colicSklearn():
    '''sklearn的逻辑回归函数
    Parameters：无
    Returns：正确率
    Author：Li Wei
    '''
    frTrain = open('机器学习实战/Ch05/horseColicTraining.txt')                                        #打开训练集
    frTest = open('机器学习实战/Ch05/horseColicTest.txt')                                                #打开测试集
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