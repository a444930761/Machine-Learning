# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:12:19 2018

@author: Administrator
"""

def loadDataSet():
    '''设置测试数据
    Parameters：无
    Returns：测试数据
    Author：Li Wei
    '''
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    '''挑选出数据集中所有的元素
    Parameters：数据集
    Returns：所有唯一元素的集合
    Author：Li Wei
    '''
    C1 = []
    for tranaction in dataSet:
        for item in tranaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1)) 
    #frozenset生成冻结集合，该集合不可更改，set生成的可以更改
    #因为下面要将集合的各个项作为字典键使用，因此这里使用冰冻集合

def scanD(D,Ck,minSupport):
    '''计算每个项集的支持度，并生成满足最小支持度的项集列表
    Parameters：D - 项集的集合
                Ck - 子项集的集合
                minSupport - 最小支持度
    Returns：retList - 满足最小支持度的项集列表
             supportDate - 项集支持度
    Author：Li Wei
    '''
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid): #判断can集合是不是tid集合的子集
                try:
                    ssCnt[can] += 1
                except:
                    ssCnt[can] = 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0,key) #在索引0的位置插入，只是为了是列表看起来有组织，所以采用这种插入
        supportData[key] = support
    return retList,supportData
'''
dataSet = loadDataSet()
C1 = createC1(dataSet)
D = list(map(set,dataSet))
L1,suppData0 = scanD(D,C1,0.5)
'''
def aprioriGen(Lk,k):
    '''拼接函数
    Parameters：Lk - 满足最小支持度的项集集合
                k - 要合成的项集大小(含几个元素)
    Returns：retList - 集合后的项集列表
    Author：Li Wei
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2: #r如果两个项集的前k-2项的元素相等，则将这两个集合合成大小为k的集合
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    '''生成候选项集函数
    Parameters： dataSet - 初始数据集
                 minSupport - 最小支持度
    Returns： L - 所有候选项集列表
              supportData - 候选项集支持度
    Author：Li Wei
    '''
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK) #将supK中的键值添加到supportData中
        L.append(Lk)
        k += 1
    return L,supportData

def rulesFromConseq(freqSet,H,suppotData,br1,minConf=0.7):
    '''针对每个项的元素超过2个的项集进行调整
    Parameters： freqSet - 项集
                 H - 项集中的子项集列表(可引出的项集)
                 supportData - 每个项的支持度字典
                 br1 - 符合最小置信度的项集列表
                 minConf - 最小置信度
    Returns： 无
    Author：Li Wei
    '''
    m = len(H[0]) #子项集的长度
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H,m + 1)
        Hmp1 = calcConf(freqSet,Hmp1,suppotData,br1,minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,suppotData,br1,minConf)

def calcConf(freqSet,H,supportData,br1,minConf=0.7):
    '''针对每个项只有2个元素的项集求置信度
    Parameters：freqSet - 项
                H - 项所包含的子项集(2个元素的项集，子项集就是元素)
                supportData - 所有项的支持度字典
                br1 - 符合置信度的项集列表
                minConf - 最小置信度
    Returns：prunedH - 满足最小置信度的元素列表
    Author：Li Wei
    '''
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            br1.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def generateRules(L,supportData,minConf=0.7):
    '''主函数，调用候选项集生成函数和置信度评估函数
    Parameters： L - 所有候选项集数据集
                 supportData - 支持度
                 minConf - 最小置信度
    Returns：符合规则的候选项集
    Author：Li Wei
    '''
    bigRuleList = []
    for i in range(1,len(L)): #因为L[0]是单个元素项集，无法构建关联规则，所以从L[1]开始，即每个项集的元素个数大于等于2开始
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] #创建只包含单个元素的结合列表
            if (i > 1): #项集的元素项目超过2个，用rulesFromseq进行合并
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else: #如果项集中只有两个元素，使用calcConf计算置信度
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

'''
rules = generateRules(L,suppData,minConf=0.7)
'''

'''毒蘑菇案例'''
mushDatSet = [line.split() for line in open('Ch11/mushroom.dat').readlines()]
L,suppData = apriori(mushDatSet,minSupport=0.3)