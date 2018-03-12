# -*- coding: utf-8 -*-
'''
决策树是最容易理解和看懂的一个模型，
通俗的讲，决策树就是if...then逻辑
针对所有的属性，进行if..then判断，
最终每个属性值下面所涵盖的都是同一类的，则结束。
属性的先后顺序通过信息增益、信息增益率或者基尼系数进行选择
信息增益的计算公式：
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def calcShannonEnt(dataSet):
    '''计算香农熵
    Parameters：dataSet - 要计算的数据集
    Returns：shannonEnt - 香农熵值
    Author：Li Wei
    '''
    m = len(dataSet)
    labelCount = {}
    for i in dataSet:
        currentlabel = i[-1]
        labelCount[currentlabel] = labelCount.get(currentlabel,0) + 1
    shannonEnt = 0
    for key in labelCount:
        prob = float(labelCount[key])/m
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt

def createDataSet():
    '''构建示例数据集
    Parameters：无
    Returns：dataSet - 数据集
             labels - 数据集标签
    Author：Li Wei
    '''
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#dataSet,labels = createDataSet()
#calcShannonEnt(dataSet)
#熵值越高，则混合的数据类型也越多
    
def splitDataSet(dataSet,axis,value):
    '''分割函数
    Parameters：dataSet - 要分割的数据集
                axis - 要分割的特征索引
                value - 要分割的特征值
    Returns：retDataSet - 分割后的数据
    Author：Li Wei
    '''
    retDataSet = []
    for i in dataSet:
        if i[axis] == value:
            data = i[:axis] + i[axis+1:]
            retDataSet.append(data)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''寻找最有特征函数
    Parameters：dataSet - 数据集
    Returns：bestFeature - 最有特征的索引值(信息增益最大的)
    Author：Li Wei
    '''
    n = len(dataSet[0]) - 1 #特征数，减一是因为最后一列是标签
    baseEntripy = calcShannonEnt(dataSet) #计算数据集的信息增益
    bestInfoGain = 0
    bestFeature = -1
    for i in range(n):
        datavaluelist = [example[i] for example in dataSet]
        datavalue = set(datavaluelist)
        newEntripy = 0
        for value in datavalue:
            retDataSet = splitDataSet(dataSet,i,value)
            prob = len(retDataSet) / float(len(dataSet))
            newEntripy += prob * calcShannonEnt(retDataSet)
        infoGain = baseEntripy - newEntripy
        print('第{}个特征的增益为{}'.format(i,infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#chooseBestFeatureToSplit(dataSet)

        
#构建树的最优特征已经挑选出来了，下面进行树的构建
def majorityCnt(classList):
    '''筛选最多类标签函数(当只有最后一个特征，且标签不完全相同时，筛选出标签做多的作为结果)
    Parameters：classList - 标签数据
    Returns：出现次数最多的标签
    Author：Li Wei
    '''
    classCount = {}
    for i in classList:
        classCount[i] = classCount.get(i,0) + 1
    sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels,featlabels):
    '''构建树函数
    Parameters：dataSet - 数据集
                labels - 数据集对应的标签
                featlabels - 最优特征索引
    Returns：myTree - 决策树
    Author：Li Wei
    '''
    classList = [example[-1] for example in dataSet]
    if len(set(classList)) == 1:
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    featlabels.append(bestFeatureLabel) #将每次选取的最优特征存储起来，用树进行验证的时候会用到
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
#        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature
              ,value),labels,featlabels)
    return myTree

#featlabels=[] #将每次选取的最优特征存储起来，用树进行验证的时候会用到
#createTree(dataSet,labels,featlabels)
    
#决策树可视化
def getNumLeafs(myTree):
    '''获取决策树叶子节点的数目
    Parameters：myTree - 决策树
    Returns：numLeafs - 决策树的叶子节点的数目
    Author：Li Wei
    '''
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    '''获取决策树的层数函数
    Parameters：myTree - 决策树
    Returns： maxDepth - 决策树的层数
    Author：Li Wei
    '''
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt,centerpt,parentpt,nodetype):
    '''绘制结点函数
    Parameters：nodeTxt - 结点名
                centerpt - 文本位置
                parentpt - 标注的箭头位置
                nodetype - 结点格式
    Returns：无
    Author：Li Wei
    '''
    arrow_args = dict(arrowstyle='<-') #定义箭头格式
    font = FontProperties(fname='c:/windows/fonts/msyhl.ttc',size=14) #设置中文字体
    createPlot.ax1.annotate(nodeTxt,xy=parentpt,xycoords='axes fraction',xytext=centerpt,
                            textcoords='axes fraction',va='center',ha='center',bbox=nodetype,
                            arrowprops=arrow_args,FontProperties=font) #绘制节点

def plotMidText(cntrpt,parentpt,txtstring):
    '''标注有向边属性值
    Parameters：cntrpt、parentpt - 计算标注位置
                txtstring - 标注的内容
    Returns：无
    Author：Li Wei
    '''
    xMid = (parentpt[0] - cntrpt[0]) / 2 + cntrpt[0]
    yMid = (parentpt[0] - cntrpt[1]) / 2 + cntrpt[1]
    createPlot.ax1.text(xMid,yMid,txtstring,va='center',ha='center',
                        rotation=30)

def plotTree(myTree,parentpt,nodeTxt):
    '''绘制决策树
    Parameters：myTree - 决策树
                parentpt - 标注的内容
                nodeTxt - 节点名
    Returns：无
    '''
    decisionNode = dict(boxstyle='sawtooth',fc='0.8')
    leafNode = dict(boxstyle='round4',fc='0.8')
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrpt = (plotTree.xOff + (1 + float(numLeafs))/2/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrpt,parentpt,nodeTxt)
    plotNode(firstStr,cntrpt,parentpt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrpt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrpt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrpt,str(key))
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD
    
def createPlot(inTree):
    '''创建绘制面板
    Parameters：inTree - 决策树
    Returns：无
    Author：Li Wei
    '''
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1
    plotTree(inTree,(0.5,1),'')
    plt.show()
    

#模型创建结束，下面应用到新的数据上
def classify(inputTree,featlabels,testvec):
    '''模型树的应用函数
    Parameters：inputTree - 树模型
                featlabels - 模型树的最优特征
                testvec - 测试数据
    Returns：classLabel - 测试数据的标签
    Author：Li Wei
    '''
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featlabels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featlabels,testvec)
            else:
                classLabel = secondDict[key]
    return classLabel

#决策树的存储
import pickle

def storTree(inputTree,filename):
    '''存储决策树函数
    Parameters：inputTree - 生成的决策树
                filename - 决策树的存储文件名
    Returns：无
    Author：Li Wei
    '''
    with open(filename,'wb') as f:
        pickle.dump(inputTree,f)
        
def grabTree(filename):
    '''读取决策树函数
    Parameters：filename - 决策树的存储路径
    Returns：决策树
    Author：Li Wei
    '''
    f = open(filename,'rb')
    return pickle.load(f)

if __name__ == '__main__':
    dataSet,labels = createDataSet()
    featlabels = []
    myTree = createTree(dataSet,labels,featlabels)
    storTree(myTree,'classTree.txt')
    testVec = [1,0]
    result = classify(myTree,featlabels,testVec)
    print(result)
    


'''使用sklearn库实现决策树'''
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.externals.six import StringIO

file = '机器学习实战/Ch03/lenses.txt'

def loadfile(filename):
    '''读取文件函数
    Parameters：filename - 文件目录
    Returns：dataSet - 数据集
    Author：Li Wei
    '''
    f = open(file)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate','labels']
    data = pd.read_table(f,names=lensesLabels) #因为文件名含有中文，直接读取报错，所以通过这种方式加载
    f.close()
    dataSet = data.iloc[:,:-1] #加载数据集
    labels = data.iloc[:,-1] #加载标签
    #因为数据集都是字符串格式，需编译成数字
    le = LabelEncoder()
    for col in dataSet.columns:
        dataSet.loc[:,col] = le.fit_transform(dataSet[col])
    return dataSet,labels

if __name__ == '__main__':
    dataSet,labels = loadfile(file)
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(dataSet.values.tolist(),labels)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,
                         feature_names = dataSet.keys(),
                         class_names = clf.classes_,
                         filled=True,rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')
    