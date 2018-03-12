# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:43:48 2017

@author: Administrator
"""
'''
算法的理论知识，参考如下
http://blog.csdn.net/acdreamers/article/details/44661149
'''
from math import log
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

def calcShannonEnt(dataSet):#定义求熵函数
    numEntries = len(dataSet)#总共有多少条数据
    labelCounts = {}#建立一个字典，收集结果中的分类
    for i in dataSet:
        currentLabel = i[-1] #注意数据集的分类要在最后一列
        if currentLabel not in labelCounts.keys():#统计各分类数量
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:#计算系统的信息熵
        prob = float(labelCounts[key])/numEntries#计算每种类别的概率
        shannonEnt -= prob*log(prob,2)#计算所有信息期望值的和即为信息熵
        #这一句可理解成这样shannonEnt += -prob*log(prob,2)
    return shannonEnt

def createDataSet():#定义创建数据集函数
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#myDat,labels = createDataSet()
#print(calcShannonEnt(myDat)) #目前结果有两类，可得出信息熵是0.97，我们增加一个分类
#myDat[0][-1] = 'maybe'
#print(calcShannonEnt(myDat)) #可以看到信息熵增加了，也就是说，数据越无序(即越不可预测)，熵越大

def splitDataSet(dataSet,axis,value):#按照给定特征划分数据集
    #axis为dataSet列方向的索引(某个特征)，value为该列(特征)所包含的值(类别)
    #这个函数的目的是将dataSet按照某列的某值进行分类
    retDataSet = []
    for i in dataSet:
        if i[axis] == value:
            reducedFeatvec = i[:axis]
            reducedFeatvec.extend(i[axis+1:])
            retDataSet.append(reducedFeatvec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    #该函数的目的是选择最好的数据集划分特征
    numFeatures = len(dataSet[0])-1 #获得数据集中的特征个数
    #选出一个元素-1是因为最后一列是标签，不是特征1
    #这里的前提是，数据集的最后一列一定是类别标签
    baseEntropy = calcShannonEnt(dataSet)#计算最初的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):#按特征的个数进行循环
        featList = [example[i] for example in dataSet]#获取索引为i的特征列表
        uniqueVals = set(featList) #获取特征所包含的唯一值
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))#划分后的各分支所在比例
            newEntropy += prob*calcShannonEnt(subDataSet)#划分后的信息熵
        infoGain = baseEntropy-newEntropy#最初的熵和划分后的熵之差
        if (infoGain>bestInfoGain):#选择熵之差最大的一列，获取其索引
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    #该函数的作用是，当所有的特征都处理完了以后，标签仍然不存在唯一值，这个时候我们
    #人为选择出现标签出现最多的那个
    classCount = {} #定义标签字典
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),
                                  reverse = True)
        return sortedClassCount[0][0]

def createTree(dataSet,labels):#创建树函数
    classList = [example[-1] for example in dataSet] #将所有的标签提取出来，组成一个列表
    if classList.count(classList[0]) == len(classList):#判断标签列表时不是唯一值
        return classList[0]
#    if set(classList) == 1:
#        return classList[0]
    if len(dataSet[0]) == 1:#如果标签不唯一，但dataSet的长度已经为1了，即所有的特征已经
        #处理完了，这个时候，认为选择标签出现最多的那一个
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)#获取最大信息熵的特征索引
    bestFeatLabel = labels[bestFeat]#获取对应的标签
    myTree = {bestFeatLabel:{}} #建立树的字典
    del labels[bestFeat] #从标签列表中删除已经处理过的特征标签
    featValues = [example[bestFeat] for example in dataSet] #获取筛选出的特征里的所有值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),
              subLabels) #递归使用cerateTree函数，直到标签值唯一或者所有特征处理完为止
        #每一次递归都会将筛选出的拥有最大信息熵的特征传递给myTree字典，以构建树
    return myTree 
    
myDat,labels = createDataSet()
myTree = createTree(myDat,labels)
#输出结果为{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
#解释如下，首先通过最大信息熵选择第一个特征，即‘no surfacing’，它下面有两个值，
#分别是0，1。0分支下面所有的标签都是no，即类标签，那么该节点判断完毕，为叶子结点
#1分支下面还含有其他特征，继续通过最大信息熵进行选择，这次选择的是特征是'flippers'，
#它下面有两个值，0和1，每个值下面都是类标签，所以该节点判断完毕，为叶子节点


'''
初版
def ceratePlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    ceratePlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(ceratePlot.ax1,'a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(ceratePlot.ax1,'a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''

def classify(inputtree,featLabels,testvec):#定义测试决策树函数
    #后两个参数代表的是指定标签以及对应标签的值
    firstStr = list(inputtree.keys())[0]
    secondDict = inputtree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testvec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key],featLabels,testvec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputtree,filename):#定义一个函数，保存决策树结果
    import pickle
    f = open(filename,'wb')
    pickle.dump(inputtree,f)
    f.close()
    
def grabTree(filename):#定义一个函数，读取决策树结果
    import pickle
    with open(filename,'rb') as f:
        return pickle.load(f)
    
    

#画图的时候要确定图所需大概空间，通过以下函数来定义相关属性    
def getNumLeafs(myTree):#定义一个获取叶节点数目的函数
    numLeafs = 0
    firstStr = list(myTree.keys())[0] #第一层肯定只有一个特征
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            maxDepth = 1 + getTreeDepth(secondDict[key])
        else:
            maxDepth = 1 #这里不是+=1，是因为这里是在节点的循环里，层数与节点数无关
    return maxDepth
       

decisionNode = dict(boxstyle='sawtooth',fc='0.8')#设置文本外框样式
leafNode = {'boxstyle':'round4','fc':'0.8'}#设置文本外框样式
arrow_args={'arrowstyle':'<-'} #注意这里是反箭头
def plotNode(nodeTxt,centerpt,parentpt,nodeType):
    axes.annotate(nodeTxt,xy=parentpt,xycoords='axes fraction',
                            xytext=centerpt,textcoords='axes fraction',va='center',
                            ha='center',bbox=nodeType,arrowprops=arrow_args)
    #str=nodeTxt设置要插入的文本，xytext为文本及箭头起始，xy为箭头终止位置，因为通过
    #arrowprops设置箭头类型为反方向的，因此箭头实际指向起始点即文本的位置
    #xycoords及textcoords是设置坐标系标准的，'axes fraction'代表
   # 0,0 是轴域左下角，1,1 是右上角，bbox中的boxstyle属性为设置文本外框样式
   #annotate的详解请参考matplotlib文档或者以下博文http://blog.csdn.net/wizardforcel/article/details/54782628
   
def plotMidText(cntrpt,parentpt,txtString):#该函数的作用是在两个节点之间添加文本
    xMid = (parentpt[0]-cntrpt[0])/2 + cntrpt[0] 
    yMid = (parentpt[1]-cntrpt[1])/2 + cntrpt[1]
    axes.text(xMid,yMid,txtString)#text是在指定的位置添加文本
    
def plotTree(myTree,parentpt,nodeTxt):
    global xoff,yoff
    numLeafs = getNumLeafs(myTree) #因为这个函数要迭代，
    #所以这里重新定义一个获取叶子节点个数的变量
#    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrpt = (xoff+(1+float(numLeafs))/(2*totalW),yoff)
    #这里首先要理解一个前提，就是为了确定整体图形比较对称，当前节点所在的位置，
    #是由他拥有的叶子节点数来决定的。比如该节点下面共有叶子节点m个，那么这m个叶子节点
    #所占x轴的长度就是m*(1/n)，当前节点要居中，那长度就是(1/2)*(m*(1/n))，
    #(叶子结点的位置需要进行偏移，但节点不需要)所以最初有个偏移所以要加回来，
    #即(1/2)*(m*(1/n))+(1/2)*(1/n)，合并后即为(1+m)/(2*n),此时的位置为相对位置
    #最后再加上根据已画叶子节点产生的偏移量xoff，就是当前节点最终在x轴上的位置
    plotMidText(cntrpt,parentpt,nodeTxt)
    plotNode(firstStr,cntrpt,parentpt,decisionNode)#画当前节点和上层节点的连接线
    secondDict = myTree[firstStr]
    yoff = yoff - 1/totalD #y轴的位置，依据深度进行递减
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            plotTree(secondDict[key],cntrpt,str(key))
        else:
            xoff = xoff + 1/totalW #每画一个叶子节点，偏移量就要加上对应的长度
            plotNode(secondDict[key],(xoff,yoff),cntrpt,leafNode)
            plotMidText((xoff,yoff),cntrpt,str(key))
    yoff = yoff + 1/totalD #注意这里，因为上面的for循环中是隶属于某个节点的，比如同层
    #有节点A和节点B，首先在节点A里面进行迭代画图，每次迭代yoff都会减1/totalD,因此，当
    #A节点结束要处理B节点的时候，yoff的值要相应加回来
        


#定义一个数信息，测试以上代码
def retrieveTree(i):
    listoftrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listoftrees[i]

inTree = retrieveTree(0)

fig=plt.figure(1,facecolor='white')
fig.clf()
axprops = dict(xticks=[],yticks=[])#这两个参数设置xy轴不显示
axes = plt.subplot(111,frameon=False,**axprops)
totalW = float(getNumLeafs(inTree)) #获取总的叶子结点个数
totalD = float(getTreeDepth(inTree)) #获取总的深度
xoff = -1/(2*totalW) #设定x轴上的偏移量，假定树一共有n个叶子节点，将x轴等分为n个段，
#每段的长度就是1/n，为了对称画图，每个叶子节点在每段的中间位置，所以要整体往左偏移
#即减去(1/2)*(1/n)，最初偏移量就是-1/(2*n)，那么第i个节点的x的位置就是i*(1/n)-1/(2*n)
#即xoff + i*(1/n)，xoff这个变量后面会根据已画的叶子节点数变化
yoff = 1
plotTree(inTree,(0.5,1),'')#这里设定0.5，1是因为plotTree函数里面有两个功能，画当前节点
#以及和上层节点的连接线，第一层节点没有上一层，所以设定这个值，使得当前节点和上层节点
#画在一起
plt.show()    

'''对隐形眼镜进行决策分类'''
data = pd.read_table('D:/Anaconda/test/lenses.txt',sep='\t',header=None)
data_matrix = data.as_matrix()
datalist = data_matrix.tolist()
labelslist = ['age','prescript','astigmatic','tearRate']
inTree = createTree(datalist,labelslist)
plotTree(inTree,(0.5,1),'')
plt.show() 


'''使用sklearn库构建决策树'''
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
labelslist = ['age','prescript','astigmatic','tearRate']
traingdata = []
trainglabel = []
with open('D:/Anaconda/test/lenses.txt','r') as f:
    for i in f.readlines():
        ilist = i.strip().split('\t')
        traingdata.append(ilist[:-1])
        trainglabel.append(ilist[-1])
        
le = LabelEncoder()
traingdata = np.array(traingdata)
for i in range((traingdata.shape)[1]):
    traingdata[:,i] = le.fit_transform(traingdata[:,i])
    #采用sklearn的LabelEncoder包将字符转换为数值
    
sktree = tree.DecisionTreeClassifier()
sktree.fit(traingdata,trainglabel)

'''绘图一'''
with open('tree.dot','w') as f:
    f = tree.export_graphviz(decision_tree=sktree,out_file=f,feature_names=labelslist)
    
'''绘图2'''
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(sktree, out_file = dot_data,                            
                        feature_names = labelslist,
                        filled=True, rounded=True,
                        special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("tree.pdf")   
