# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:54:36 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def loadDataSet(file):
    '''加载数据函数
    Parameters：file - 存放数据的文件
    Returns；dataMat - 数据列表
    Author：Li Wei
    '''
    dataMat =[]
    f = open(file)
    for i in f.readlines():
        ilist = i.strip().split('\t')
        fltline = list(map(float,ilist))
        dataMat.append(fltline)
    return dataMat

def distEclud(vecA,vecB):
    '''计算两向量之间距离函数（欧氏距离）
    Parameters：vecA - 向量A
                vecB - 向量B
    Returns：两向量之间的距离
    Author：Li Wei
    '''
    return np.sqrt(np.sum(np.power((vecA - vecB),2)))

def randCent(dataSet,k):
    '''选定k个聚类中心函数
    Parameters：dataSet - 所操作的数据集
                k - 需要聚类的质心数
    Returns：   centroids - k个质心
    Author：Li Wei
    '''
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k,n)))
    for i in range(n):
        mini = dataSet[:,i].min()
        rangei = float(dataSet[:,i].max() - mini)
        centroids[:,i] = mini + np.random.rand(k,1) * rangei
    return centroids

file = 'Ch10/testSet.txt'
dataMat = np.mat(loadDataSet(file))
plt.scatter(dataMat.A[:,0],dataMat.A[:,1]) #将原数据画出来
label0 = mlines.Line2D([],[],color='r',marker='.',markersize=6,label='类别0')
plt.legend(handles=[label0])

def  kMeans(dataSet,k,distMeas=distEclud,creatCent=randCent):
    '''kMeans算法函数
    Parameters：dataSet - 要操作的数据集
                k - 要聚类的质心数量
                distMeas - 计算距离方法
                creatCent - 选择质心方法
    Returns: centroids - 寻找的质心
             clusterAssment - 分类标签
    Author：Li Wei
    '''
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = creatCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                distJI = distMeas(dataSet[i,:],centroids[j,:]) #计算每个点到质心的距离
                if distJI < minDist: #
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex: #下面重新计算质心位置以后，所有的点都会进行
                #重新分配，如果重新分配的结果有改变，则继续循环，直到不再改变
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #记录质心的索引及距离
            #距离之和(SSE,误差平方和)，度量聚类效果的指标，SSE越小，效果越好
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust,axis=0) #通过计算每个簇的距离均值来更改质心
    return centroids,clusterAssment

myCentroids,clustAssing = kMeans(dataMat,4)
#通过输出可以看到程序经过8次循环达到收敛,次数与初始点的选择有关系

'''将分类结果画出来'''
dataMat_np = dataMat.A
clustAssing_np = clustAssing.A
dataMat_np = np.column_stack((dataMat_np,clustAssing_np[:,0]))

import matplotlib.lines as mlines
clist = []
for i in range(4):
    if i == 0:
        clist.append('r')
    if i == 1:
        clist.append('b')
    if i == 2:
        clist.append('y')
    if i == 3:
        clist.append('g')
label0 = mlines.Line2D([],[],color='r',marker='.',markersize=6,label='类别0')
label1 = mlines.Line2D([],[],color='b',marker='.',markersize=6,label='类别1')
label2 = mlines.Line2D([],[],color='y',marker='.',markersize=6,label='类别2')
label3 = mlines.Line2D([],[],color='g',marker='.',markersize=6,label='类别3')
fig = plt.figure()
#data = dataMat_np[dataMat_np[:,2] == 0]
plt.scatter(dataMat_np[:,0],dataMat_np[:,1],c=clist,s=20,alpha=0.5)
plt.legend(handles=[label0,label1,label2,label3])
plt.show()
    

'''使用后处理来提高聚类性能
1、将具有最大SSE值的簇划分成两个簇
2、为了保持簇总数不变，可以将某两个簇进行合并
2.1、合并最近的质心(计算两质心之间的距离，合并距离最小的)
2.2、合并两个使SSE增幅最小的质心(两两合并，计算总SSE值，找增幅最小的)
'''
'''二分k-均值算法
1、将所有的点作为一个簇
2、将该簇一分为二
3、如果簇的个数小于k，依次对2中的簇进行划分，最终选择SSE降幅最大的或者SSE最大的继续划分
4、重复2，3直到等于k
'''
def biKmeans(dataSet,k,distMeas=distEclud):
    '''二分k-均值聚类算法
    Parameters：dataSet - 要聚类的数据
                k - 簇的个数
                distMeas - 距离算法
    Returns：centList - 质心
             clusterAssment - 分类后标签及误差
    Author：Li Wei
    '''
    m = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid = np.mean(dataSet,axis=0).tolist()[0] #构建初始质心，为每个特征的均值
    centList = [centroid] #质心列表
    for i in range(m):
        clusterAssment[i,1] = distMeas(np.mat(centroid),dataSet[i,:]) ** 2
    while (len(centList)<k):
        lowestSSE = float('inf')
        for i in range(len(centList)): #迭代将每个簇都进行划分
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i )[0],:] #筛选出当前簇的所有数据
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)#获取划分后的质心及标签
            sseSplit = np.sum(splitClustAss[:,1])#计算划分后的SSE总值
            sseNoSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0],1]) #计算没筛选的SSE总值
            print('sseSplit,and notSplit',sseSplit,sseNoSplit)
            if (sseSplit + sseNoSplit) < lowestSSE: #两者如果小于最初的SSE
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNoSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #设为新的簇标签
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit #将原来的簇标签替换掉
        print('the bestCentToSplit is:',bestCentToSplit)
        print('the len of bestClustAss is:',len(bestClustAss))
        #划分后，用其中的一个质心替换原来的质心，并添加另一个的质心
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss #更新原来的标签表
    return centList,clusterAssment

file2 = 'Ch10/testSet2.txt'
data2_np = np.array(loadDataSet(file2))
plt.scatter(data2_np[:,0],data2_np[:,1],s=20)
datMat2 = np.mat(data2_np)
centList,myNewAssments = biKmeans(datMat2,3)


'''对地理坐标进行聚类'''
def distSLC(vecA,vecB):
    '''计算经纬度距离的函数，曲面的，不能直接相加减
    Parameters：vecA - 点A
                vecB - 点B
    Returns：两点的球面距离
    Author：Li Wei
    '''
    a = np.sin(vecA[0,1] * np.pi/180) * np.sin(vecB[0,1] * np.pi/180)
    b = np.cos(vecA[0,1] * np.pi/180) * np.cos(vecB[0,1] * np.pi/180) * \
                                np.cos(np.pi * (vecB[0,0] - vecA[0,0])/180)
    return np.arccos(a+b) * 6371

def clusterClubs(numClust=5):
    '''将地图上的点进行聚类并绘制出来
    Parameters：numClust - 要聚类的个数
    Returns：无
    Author：Li Wei
    '''
    file = 'Ch10/places.txt'
    datList = []
    f = open(file)
    for line in f.readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
    f.close()
    datMat = np.mat(datList)
    myCentroids,clustAssing = biKmeans(datMat,numClust,distMeas=distSLC)
    
        
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8] #设置绘图区域与画布的位置关系百分比(left,bottom,width,height)
    scatterMarkers=['s','o','^','8','p','d','v','h','>','<']
    axprops = dict(xticks=[],yticks=[])
    ax0 = fig.add_axes(rect,label='ax0',**axprops)
    imgP = plt.imread('Ch10/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect,label='ax1',frameon=False) #通过标签将两个绘图区区分开
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A ==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                    ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle,s=90)
        #flatten().A 将矩阵降为一维，并转化为数组
    ax1.scatter(myCentroids[:,0].flatten().A[0],\
                myCentroids[:,1].flatten().A[0],\
                marker='+',s=300)
    plt.show()