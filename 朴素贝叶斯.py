# -*- coding: utf-8 -*-
'''
朴素贝叶斯算法是有监督的学习算法，解决的是分类问题，
如客户是否流失、是否值得投资、信用等级评定等多分类问题。
该算法的优点在于简单易懂、学习效率高、在某些领域的分类问题中
能够与决策树、神经网络相媲美。但由于该算法以自变量之间的独立（条件特征独立）性
和连续变量的正态性假设为前提，就会导致算法精度在某种程度上受影响。

朴素贝叶斯的决策理论：
p1(x,y)是数据(x,y)属于1类的概率，p2(x,y)是数据属于2类的概率
p1(x,y) > p2(x,y) 则判定数据属于1类
p1(x,y) < p2(x,y) 则判定数据属于2类

条件概率公式：P(A|B) = P(B|A)P(A) / P(B)
在事件B发生的情况下，事件A发生的概率

全概率公式：P(B) = P(B|A)P(A) + P(B|A')P(A')
如果A和A'构成样本空间的一个划分，那么事件B的概率，等于A和A'的概率分别
乘以B对这两个事件的条件概率之和

对条件概率公式进行变形：P(A|B) = P(A)*(P(B|A)/P(B))
P(A)称为‘先验概率’，即在B发生前，事件A的概率
P(A|B)称为‘后验概率’，即在B发生后，事件A的概率
P(B|A)/P(B)称为‘可能性函数’，这是一个调整因子，使得预估概率更接近真实概率
所以条件概率公式可以理解为
后验概率 = 先验概率 * 调整因子

朴素贝叶斯对条件个概率分布做了独立性假设，即：
p(X|a) = p(x1|a)*p(x2|a)*p(x3|a)...p(xn|a)
'''
import numpy as np
from functools import reduce

#构建言论过滤器模型
def loadDataSet():
    '''创建数据集函数
    Parameters：无
    Returns：postingList - 数据切分的词条
             labels - 类别标签
    Author: Li Wei
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],         #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0,1,0,1,0,1] #0代表非侮辱类，1代表侮辱类
    return postingList,labels

#整理词汇表
def createVocabList(dataSet):
    '''将词条数据整理成不重复的词条列表，即词汇表
    Parameters：dataSet - 词条数据
    Returns：vocabSet - 词汇表
    Author：Li Wei
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #循环取并集
    return list(vocabSet)

#根据词汇表将词条向量化
def setOfWords2Vec(vocabList,inputSet):
    '''词条向量化函数
    Parameters：vocabList - 词汇表
                inputSet - 词条
    Returns：returnVec - 向量化后的词条
    Author：Li Wei
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word "{}" is not in vocabList '.format(word))
    return returnVec

#训练分类器
def trainNB0(trainMatrix,trainCategory):
    '''训练朴素贝叶斯分类器函数
    Parameters：trainMatrix - 向量化后的词条
                trainCategory - 数据标签
    Returns：p0Vect - 非侮辱类的条件概率数组
             P1Vect - 侮辱类的条件概率数组
             pAbusive - 文档属于侮辱类的概率
    Author：Li Wei
    '''
    numTrainDocs = len(trainMatrix) #训练的词条数目
    numWords = len(trainMatrix[0]) #每个词条的单词数(向量化后统一为词汇表的长度)
    pAbsuive = sum(trainCategory) / float(numTrainDocs) #词条属于侮辱类的概率
    #侮辱类的标签是1，非侮辱类的标签是0，所以sum(trainCategory)的结果是侮辱类的和
    p0Num = np.ones(numWords) ; p1Num = np.ones(numWords)
    p0Denom = 2 ; p1Denom = 2  
    #这里，采用拉普拉斯平滑将分子初始化为1，分母初始化为2，
    #是为了避免当出现概率为0的单词后，整个词条的概率为0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #统计属于侮辱类的条件概率所需的数据，即即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else: 
            p0Num += trainMatrix[i] #统计属于非侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect,p1Vect,pAbsuive
    
#开始分类
def classify(vecdata,p0Vec,p1Vec,pClass1):
    '''朴素贝叶斯分类函数
    Parameters：vecdata - 待分类的词条数组
                p0Vec - 侮辱类的条件概率数组
                p1Vec - 非侮辱类的条件概率数组
                pClass1 - 文档属于侮辱类的概率
    Returns：1 - 属于侮辱类
             0 - 属于非侮辱类
    Author：Li Wei
    '''      
    #p1 = reduce(lambda x,y:x*y,vecdata*p1Vec) * pClass1
    #p0 = reduce(lambda x,y:x*y,vecdata*p0Vec) * (1-pClass1)
    p1 = sum(vecdata*p1Vec) + np.log(pClass1)
    p0 = sum(vecdata*p0Vec) + np.log(1-pClass1)
    #这里采用对数是为了避免小数相乘太小出现下溢的情况
    if p1 > p0:
        return 1
    else:
        return 0
    
#测试
def testingNB(testEntry):
    '''测试朴素贝叶斯分类器
    Parameters：testEntry - 要测试的词条
    Returns：无
    Author：Li Wei
    '''
    dataSet,labels = loadDataSet()
    myVocabList = createVocabList(dataSet)
    trainMat = []
    for data in dataSet:
        trainMat.append(setOfWords2Vec(myVocabList,data))
    p0V,p1V,pAb = trainNB0(trainMat,labels)
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    if classify(thisDoc,p0V,p1V,pAb): #为1是侮辱，0是非侮辱
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
   
'''
if __name__ == '__main__':
    testEntry = input('输入词条：')
    testEntry = testEntry.split()
    testingNB(testEntry)
'''
    
'''朴素贝叶斯之过滤垃圾邮件'''
import re

def textParse(bigString):
    '''将词条转化为单词列表
    Parameters：bigString - 词条
    Returns：单词列表
    Author：Li Wei
    '''
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def bagOfWords2VecMN(vocabList,inputSet):
    '''构建词袋模型(不仅记录单词是否出现，还计算出现的次数)
    Parameters：vocabList - 词汇列表，createVocabList返回的列表
                inputSet - 切分的词条列表
    Returns：returnVec - 词条向量，词袋模型
    Author：Li Wei
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index[word]] += 1
    return returnVec

#测试
def spamTest():
    '''通过现有数据测试模型的错误率
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    docList = [] ; classList = [] ; fullText = []
    for i in range(1,26):
        wordList = textParse(open('机器学习实战/Ch04/email/spam/{}.txt'.format(i),'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('机器学习实战/Ch04/email/ham/{}.txt'.format(i),'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)) ; testSet = [] #一共50封邮件
    for i in range(10): #s随机挑选10封作为验证集
        randIndex = int(np.random.uniform(0,len(trainingSet))) #随机生成一个邮件索引
        testSet.append(trainingSet[randIndex]) #将邮件加入验证集
        del(trainingSet[randIndex]) #从训练集中删除
    trainMat = [] ; traingClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        traingClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(traingClasses)) #利用训练数据生成相关概率
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classify(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#如果验证结果和实际结果不一致
            errorCount += 1
            print('分类错误的测试集：',docList[docIndex])
    print('错误率：{:%}'.format(float(errorCount)/len(testSet)))

if __name__ == '__main__':
    spamTest()
    
'''朴素贝叶斯之新浪新闻分类'''

import os
import jieba

def TextProcessing(folder_path,test_size=0.2):
    '''读取文件构建训练集及测试集数据
    Parameters：folder_path - 文件目录
                test_size - 测试数据的比例
    Returns：all_words_list - 词汇表
             train_data_list - 分割后训练集的新闻列表
             test_data_list - 分割后的测试集的新闻列表
             train_class_list - 分割后的训练集新闻对应的类别
             test_class_list - 分割后的测试集新闻对应的类别
    Author：Li Wei
    '''
    folder_list = os.listdir(folder_path)
    data_list = [] ; class_list = []
    
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        
        j = 1
        for file in files:
            if j > 100: #最多读取100个文件
                break
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f:
                raw = f.read()
                
            word_cut = jieba.cut(raw,cut_all=False) #采用精简模式，返回一个迭代器
            word_list = list(word_cut) #将迭代器转化Wie列表
            
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    
    #分成训练集和测试集        
    data_class_list = list(zip(data_list,class_list)) #将数据和对应的标签压缩在一起
    np.random.shuffle(data_class_list) #将数据随机打乱
    index = int(len(data_class_list) * test_size) + 1 
    train_list = data_class_list[index:] #训练数据集
    test_list = data_class_list[:index] #测试数据集
    train_data_list,train_class_list = zip(*train_list) #训练集压缩
    test_data_list,test_class_list = zip(*test_list) #测试集压缩
    
    #统计词频，并排序
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            all_words_dict[word] = all_words_dict.get(word,1) + 1
    all_words_tuple_list = sorted(all_words_dict.items(),key=lambda x:x[1],reverse=True)
    all_words_list,all_words_nums = zip(*all_words_tuple_list)    #解压缩
    all_words_list = list(all_words_list)
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list 

def MakeWordsSet(words_file):
    '''整理特定词语
    Parameters：words_file - 文件路径
    Returns：words_set - 内容集合
    Author：Li Wei
    '''
    words_set = set()
    with open(words_file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set

#去除一些语气助词等之类的非特征文本
def words_dict(all_words_list,deleteN,stopwords_set=set()):
    '''选取特征文本函数
    Parameters：all_words_list - 所有文本集合
                deleteN - 要删除的词频最高的N个词
                stopwords_set - 指定的结束语
    Returns：feature_words - 最终选定的特征词
    Author：Li Wei
    '''
    feature_words = []
    n =  1
    for t in range(deleteN,len(all_words_list)):
        if n > 1000 :#挑选出1000个特征
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set \
        and 1 < len(all_words_list[t]) < 5: #非纯数字，且长度在5之内的单词保存下来
                feature_words.append(all_words_list[t])
        n += 1
    return feature_words

def TextFeatures(train_data_list,test_data_list,feature_words):
    '''向量化数据集
    Parameters：train_data_list - 训练数据集
                test_data_list - 测试数据集
                feature_words - 词汇表
    Returns：train_feature_list - 向量化后的训练数据集
             test_feature_list - 向量化后的测试数据集
    Author：Li Wei
    '''
    def text_features(text,feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text,feature_words) for text in train_data_list]
    test_feature_lsit = [text_features(text,feature_words) for text in test_data_list]
    return train_feature_list,test_feature_lsit

if __name__ == '__main__':
    folder_path = '机器学习实战/Ch04/SogouC/Sample/'
    all_words_list,train_data_list,test_data_list,train_class_list,test_class_list= TextProcessing(folder_path)
    
    stopwords_file = '机器学习实战/Ch04/SogouC/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)   
    
    feature_words = words_dict(all_words_list,100,stopwords_set)   
    trainMat,testdata = TextFeatures(train_data_list,test_data_list,feature_words)
    
'''训练集集测试集数据都已经整理完成，具体训练方法同上，
由于新闻是属于多分类问题，处理起来稍微麻烦，
下面采用sklearn实现贝叶斯方法
'''
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

def TextClassifier(traindata,testdata,trainclass,testclass):
    '''贝叶斯文本分类器
    Parameters：traindata - 训练集
                testdata - 测试集
                trainclass - 训练集标签
                testclass - 测试集标签
    Returns：test_accuracy - 分类器精度
    Author：Li Wei
    '''
    classifier = MultinomialNB().fit(traindata,trainclass)
    test_accuracy = classifier.score(testdata,testclass)
    return test_accuracy

def showaccuracy():
    '''绘制deleteN与test_accuracy的关系图函数
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    test_accuracy_list = []
    deleteNs = range(0,1000,20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list,deleteN,stopwords_set)
        train_feature_list,test_feature_list = TextFeatures(train_data_list,test_data_list,feature_words)
        test_accuracy = TextClassifier(train_feature_list,test_feature_list,train_class_list,test_class_list)
        test_accuracy_list.append(test_accuracy)
        
    plt.figure()
    plt.plot(deleteNs,test_accuracy_list)
    plt.title('train_feature_list,test_feature_list')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()    
    
if __name__ == '__main__':
    folder_path = '机器学习实战/Ch04/SogouC/Sample/'
    all_words_list,train_data_list,test_data_list,train_class_list,test_class_list= TextProcessing(folder_path)
    
    stopwords_file = '机器学习实战/Ch04/SogouC/stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
    
    #showaccuracy() #通过多次执行绘图函数发现，删除前450个，精度最高
    
    test_accuracy_list = []
    feature_words = words_dict(all_words_list, 450, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave = lambda c: sum(c) / len(c) 
 
    print(ave(test_accuracy_list))
           