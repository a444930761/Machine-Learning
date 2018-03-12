# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:38:45 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName,delim='\t'):
    '''加载数据函数
    Parameters：fileName - 数据文件，str类型
                delim - 分隔符，str类型
    Returns：加载后的数据集，矩阵类型
    Author：Li Wei
    '''
    f = open(fileName)
    stringArr = [line.strip().split(delim) for line in f.readlines()]
    f.close()
#    datArr = list(map(float,line) for line in stringArr)
    return np.mat(stringArr,dtype='float')

def pca(dataMat,topNfeat=9999999):
    '''PCA主程序
    Parameters：dataMat - 进行主成分分析的数据矩阵，矩阵类型
                topNfeat - 返回的特征数，默认前999999个，也就是全部
    Returns： lowDDataMat - 转换矩阵，矩阵类型
              reconMat - 降维之后的数据，矩阵类型
    Author：Li Wei
    '''
    meanVals = np.mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals #首先减去元素数据的均值
    covMat = np.cov(meanRemoved,rowvar=0) #计算协方差矩阵
    eigVals,eigVects = np.linalg.eig(np.mat(covMat)) #计算特征值
    eigValInd = np.argsort(eigVals) #对特征值从小到大排序，并返回对应索引
    eigValInd = eigValInd[:-(topNfeat+1):-1] #更改顺序为从大到小
    redEigVects = eigVects[:,eigValInd] #对特征按特征值从大到小排序
    lowDDataMat = meanRemoved * redEigVects #构建转换矩阵
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #将原始数据转换到新空间
    return lowDDataMat,reconMat

file = 'Ch13/testSet.txt'
dataMat = loadDataSet(file)
lowDMat,reconMat = pca(dataMat,1) #保留一个特征
lowDMat.shape 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].T.tolist()[0],dataMat[:,1].T.tolist()[0],c='b',s=20)
ax.scatter(reconMat[:,0].T.tolist()[0],reconMat[:,1].T.tolist()[0],c='r',s=20)
plt.show()

'''利用PCA对半导体制造数据降维'''
file2 = 'Ch13/secom.data'
data = loadDataSet(file2,' ')

def replaceNanWithMean(data):
    '''将Nan值转化为平均值
    Parameters：data - 数据集，矩阵类型
    Returns： data - 清理后的数据集，矩阵类型
    Author：Li Wei
    '''
    numFeat = data.shape[1]
    for i in range(numFeat):
        meanVal = np.mean(data[np.nonzero(~np.isnan(data[:,i].A))[0],i])
        
        data[np.nonzero(np.isnan(data[:,i].A))[0],i] = meanVal
        
    return data

dataMat = replaceNanWithMean(data) #将数据中的Nan值进行清理
meanVals = np.mean(dataMat,axis=0) #计算清理后的均值
meanRemoved = dataMat - meanVals #去除均值的数据
covMat = np.cov(meanRemoved,rowvar=0) #计算协方差矩阵
eigVals,eigVects = np.linalg.eig(np.mat(covMat)) #计算特征值
#可以看到eigVals中有很多的特征都是0，这意味着这些特征可以用其他特征来表示
eigplt = eigVals.cumsum()/eigVals.sum()
plt.plot(eigplt)
plt.xlim(0,20)
plt.ylabel('总方差百分比')
plt.xlabel('特征个数')
plt.show()

#以上绘图结果可看到大部分方差都包含在前几个主成分中(6个)，因此只保留6个特征就可以

