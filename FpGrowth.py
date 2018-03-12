# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:20:51 2018

@author: Administrator
"""

class treeNode:
    '''FP树中节点的类定义
    '''
    def __init__(self,nameValue,numOccur,parentNode):
        '''树节点初始化函数
        Parameters：nameValue - 树节点名字
                    numOccur - 计数值
                    nodeLink - 链接相似的元素项
                    parentNode - 当前节点的父节点
        Returns：无
        Author：Li Wei
        '''
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    
    def inc(self,numOccur):
        '''更改计数值
        Parameters：numOccur - 计数值需增加的值
        Returns：无
        Author：Li Wei
        '''
        self.count += numOccur
        
    def disp(self,ind=1):
        '''以文本形式展示树
        '''
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind + 1)

'''            
rootNode = treeNode('pyramid',9,None) #生成初始父节点
rootNode.children['eye'] = treeNode('eye',13,None) #添加子节点
rootNode.children['phoenix'] = treeNode('phoenix',3,None) #再添加一个子节点
rootNode.disp() #查看效果
'''

def updateHeader(nodeToTest,targetNode):
    '''更新节点链接函数
    Parameters：nodeToTest - 子节点，dict类型
                targetNode - 指向的示例，dict类型
    Returns：无
    Author：Li Wei
    '''
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def updateTree(items,inTree,headerTable,count):
    '''添加FP树子节点操作
    Parameters：items - 某个项集的元素排序列表，list类型
                inTree - 树，dict类型
                headerTable - 满足支持度的全局元素项(头指针)，dict类型
                count - 该项集的计数值，数值型
    Returns：无
    Author：Li Wei
    '''
    print('items:',items)
    if items[0] in inTree.children: #首先检查该项集的第一个元素是否在树中存在，存在，则增加计数值
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree) #不存在，则添加子节点
        print('name:',inTree.children[items[0]].name)
        print('parent:',inTree.children[items[0]].parent.name)
        if headerTable[items[0]][1] == None: #如果头指针中该元素的指向指针为None，则添加
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])#指向不为None，则更新
    
    if len(items) > 1 : #迭代调用添加子节点操作，每次调用去掉元素列表中的第一个元素
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)

def createTree(dataSet,minSup=1):
    '''构建FP树函数
    Parameters：dataSet - 字典化后的数据集,dict类型
                minSup - 最小支持度，数值型
    Returns：retTree - FP树，dict类型
             headerTable - 满足支持度的全局元素项(头指针)，dict类型
    Author：Li Wei
    '''
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
    klist = list(headerTable.keys())
    for k in klist:
        if headerTable[k] < minSup:
            del(headerTable[k]) #移除不满足最小支持度的元素项
    
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None,None #如果没有满足最小支持度的元素项，则退出
    
    for k in headerTable:
        headerTable[k] = [headerTable[k],None] #计数值以及指向每种类型第一个元素项的指针
    retTree = treeNode('Null Set',1,None) #创建只包含空集合的根节点
    
    for tranSet,count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),
                            key=lambda x:x[1],reverse=True)] #将频繁项集进行排序(一个项集)
            print(orderedItems)
            updateTree(orderedItems,retTree,headerTable,count) #使用排序后的频繁项集对树进行填充
    return retTree,headerTable

def loadSimpDat():
    '''构建样例数据
    Parameters：无
    Returns：simpDat - 样例数据，list类型
    Author：Li Wei
    '''
    simpDat = [list('rzhjp'),list('zyxwvuts'),['z'],list('rxnos'),
               list('yrxzqtp'),list('yzxeqstm')]
    return simpDat

def cerateInitSet(dataSet):
    '''字典化化数据
    Parameters：dataSet - 要字典化的数据，list类型
    Returns：retDict - 字典化后的数据，dict类型
    Author：Li Wei
    '''
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

simpDat = loadSimpDat()
initSet = cerateInitSet(simpDat)
myFPtree,myHeaderTab = createTree(initSet,3)

def ascendTree(leafNode,prefixPath):
    '''迭代上溯整颗树函数
    Parameters：leafNode - 树子节点，dict类型
                prefixPath - 存储前缀路径的列表，list类型
    Returns：无
    Author：Li Wei
    '''
    if leafNode.parent != None: #如果有父节点，则上溯
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):
    '''寻找前缀路径函数
    Parameters：basePat - 头指针中的单个频繁元素
                treeNode - 元素对应的链表
    Returns：condPats - 该元素的前缀路径
    Author：Li Wei
    '''
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1 :
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink #遍历链表
    return condPats

findPrefixPath('r',myHeaderTab['r'][1])

def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    '''构建频繁项集的树
    Parameters：inTree - FP树
                headerTable - 头指针
                minSup - 最小支持度
                preFix - 频繁项集
                freqItemList - 频繁项集列表
    Returns：无
    Author：Li Wei
    '''
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda x:x[1][0])] #将头指针顺序排列
    for basePat in bigL: #遍历所有头指针
        print(basePat)
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])
        print(condPattBases)
        myCondTree,myHead = createTree(condPattBases,minSup)
        
        if myHead != None:
            print('conditional tree for:',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)
        

freqItems = []
mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)   

'''从新闻网站中点击流中挖掘'''
file = 'Ch12/kosarak.dat'
parsedDat = [line.split() for line in open(file).readlines()]
parsedInit = cerateInitSet(parsedDat)
myFPtree2,myHeaderTab2 = createTree(parsedInit,100000)
myFreqList = []
mineTree(myFPtree2,myHeaderTab2,100000,set([]),myFreqList)
