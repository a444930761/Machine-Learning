# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 17:11:34 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

def loadSimpdata():
    '''构建数据集
    Parameters : 无
    Returns : dataMatrix - 数据矩阵
              classLabels - 数据标签
    Author：Li Wei          
    '''
    dataMatrix = np.matrix([[1,2.1],
                            [1.5,1.6],
                            [1.3,1],
                            [1,1],
                            [2,1]])
    classLabels = [1,1,-1,-1,1]
    return dataMatrix,classLabels

def showDataSet(dataMat,lbelMat):
    '''定义绘图函数，将数据画出来
    Parameters : dataMat - 数据集
                 lbelMat - 数据标签
    Returns ： 无
    Author：Li Wei 
    '''
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if lbelMat[i] > 0: #将正样本和负样本分开
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    

    plt.scatter(data_plus_np.T[0],data_plus_np.T[1]) #将数据集进行行列转换，将第一位设置为
    #x，第二位设置为y
    plt.scatter(data_minus_np.T[0],data_minus_np.T[1])
    plt.show()
    
def stumpClassify(dataMatrix,dim,threshVal,threshIneq):
    '''单层决策树分类函数
    Parameters：dataMatrix - 数据矩阵
                dim - 第dim列，即第几个特征
                threshVal - 阈值
                threshIneq - 类别
    Returns：retArray - 分类结果
    Author：Li Wei 
    '''
    retArray = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dim] <= threshVal] = -1 #类型为lt的，小于阈值的设为-1
    else:
        retArray[dataMatrix[:,dim] > threshVal] = -1 #类型为gt的，大于阈值的设为-1
    return retArray

def buildStump(dataArr,classLabels,D):
    '''找到最佳单层决策树
    Parameters：dataArr - 数据矩阵
                classLabels - 数据标签
                D - 样本权重
    Returns：bestStump - 最佳单层决策树信息
             minError - 最小误差
             bestClasEst - 最佳的分类结果
    Author：Li Wei 
    '''
#    m,n = dataArr.shape
    bestStump = {}
    minError = float('inf') #设置初始误差为正无穷大
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = dataMatrix.shape
    numSteps = 10.0
    bestClasEst = np.mat(np.zeros((m,1)))
    for i in range(n):
        rangeMin = dataMatrix[:,i].min() #特征最小值
        rangeMax = dataMatrix[:,i].max() #特征最大值
        stepSize = (rangeMax-rangeMin)/numSteps #步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize) #设置阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) #预测分类结果
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #预测结果和实际结果相等的，设为0
                weightedError = D.T * errArr #计算误差
                print('split: dim %d,thresh %.2f, thresh inequal: %s,the weighted error is %.3f'% (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
           
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    '''构建adaboost算法提升分类器性能
    Parameters : dataArr - 数据矩阵
                 classLabels - 数据标签
                 numIt - 循环次数
    Returns ： weakClassArr - 存储单层决策树
                aggClassEst - 分类结果
    Author：Li Wei 
    '''
    weakClassArr = [] #存储单层决策树的列表
    m = dataArr.shape[0]
    aggClassEst = np.mat(np.zeros((m,1))) #设定初始分类结果
    D = np.mat(np.ones((m,1))/m) #初始化权重
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        alpha = float(0.5 * np.log((1 - error)/max(error,1e-16))) #计算弱学习算法权重，分母这么写是为了避免当error为0的时候报错
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T,classEst)#计算e指数
        D = np.multiply(D,np.exp(expon))
        D = D / D.sum()
        
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        print('total error:',errorRate)
        if errorRate == 0:
            break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    '''分类函数
    Parameters：datToClass - 要分类的数据
                classifierArr - 训练好的分类器
    Returns： 分类结果
    Author：Li Wei 
    '''
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m,1))) #设置所有初始分类为0
    for i in classifierArr:#遍历所有单层决策树
        classEst = stumpClassify(dataMatrix,i['dim'],i['thresh'],i['ineq']) #得到输出类别值
        aggClassEst += i['alpha'] * classEst #乘以该决策树的权重，累加到aggClassEst
        print(aggClassEst)
    return np.sign(aggClassEst)

if __name__=='__main__':
    dataArr,classLabels = loadSimpdata()
#    showDataSet(dataArr,classLabels)
    weakClassArr,aggClassEst = adaBoostTrainDS(dataArr,classLabels)
    print(adaClassify([[0,0],[5,5]],weakClassArr))
    
    
'''预测病马'''
traingfile = 'Ch05/horseColicTraining.txt'
testfile = 'Ch05/horseColicTest.txt'

def getfile(file):
    '''读取文件函数
    Parameters：file - 要读取的文件
    Returns： 特征集、标签集
    Author：Li Wei 
    '''
    f = open(file)
    data = []
    for i in f.readlines():
        data.append(i.strip().split('\t'))
    data_np = np.array(data).astype(float)
    return data_np[:,:-1],data_np[:,-1]

def adaClassify2(data,classArr):
    '''预测分类函数
    Parameters：data - 要预测的数据
                classArr - 训练好的分类器
    Returns：分类结果
    Author：Li Wei 
    '''
    dataMatrix = np.mat(data)
    m = dataMatrix.shape[0]
    aggclassEst = np.mat(np.zeros((m,1)))
    for i in classArr:
        classEst = stumpClassify(dataMatrix,i['dim'],i['thresh'],i['ineq'])
        aggclassEst += i['alpha'] * classEst
    return np.sign(aggclassEst)

traingdata,trainglabels = getfile(traingfile)

trainglabels[trainglabels == 0] = -1

weakClassArr,aggClassEst = adaBoostTrainDS(traingdata,trainglabels)

testdata,testlabels = getfile(testfile)
testlabels[testlabels == 0] = -1

#errorCount = 0
#for i in range(len(testdata)):
#    classresult = adaClassify2(testdata[i,:],weakClassArr)
#    if int(classresult) !=  int(testlabels[i]):
#        errorCount += 1 

classresult1 = adaClassify2(traingdata,weakClassArr)
errorArr1 = np.mat(np.ones((len(traingdata),1)))
errRte1 = errorArr1[classresult1 != np.mat(trainglabels).T].sum()/len(traingdata)
classresult = adaClassify2(testdata,weakClassArr)
errorArr = np.mat(np.ones((len(testdata),1)))
errRate = errorArr[classresult != np.mat(testlabels).T].sum()/len(testdata)    
print('训练集错误率：',errRte1)
print('测试集错误率：',errRate)

'''使用sklearn实现AdaBoost算法'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),algorithm='SAMME',n_estimators=10)
bdt.fit(traingdata,trainglabels)
perdictions = bdt.predict(traingdata)
errArr = np.mat(np.ones((len(traingdata), 1)))
print('训练集的错误率:%.3f%%' % float(errArr[perdictions != trainglabels].sum() / len(traingdata) * 100))
predictions = bdt.predict(testdata)
errArr = np.mat(np.ones((len(testdata), 1)))
print('测试集的错误率:%.3f%%' % float(errArr[predictions != testlabels].sum() / len(testdata) * 100))


from matplotlib.font_manager import FontProperties
def plotROC(predStrengths, classLabels):
    """
    绘制ROC
    Parameters:
        predStrengths - 分类器的预测强度
        classLabels - 类别
    Returns:
        无
    Author：Li Wei 
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    cur = (1.0, 1.0)                                                         #绘制光标的位置
    ySum = 0.0                                                                 #用于计算AUC
    numPosClas = np.sum(np.array(classLabels) == 1.0)                        #统计正类的数量
    yStep = 1 / float(numPosClas)                                             #y轴步长   
    xStep = 1 / float(len(classLabels) - numPosClas)                         #x轴步长
 
    sortedIndicies = predStrengths.argsort(axis=0)                                 #预测强度排序
 
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist():
        if classLabels[index[0]] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]                                       #高度累加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b')     #绘制ROC
        cur = ((cur[0] - delX), (cur[1] - delY))                                 #更新绘制光标的位置
    ax.plot([0,1], [0,1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties = font)
    plt.xlabel('假阳率', FontProperties = font)
    plt.ylabel('真阳率', FontProperties = font)
    ax.axis([0, 1, 0, 1])
    print('AUC面积为:', ySum * xStep)                                         #计算AUC
    plt.show()