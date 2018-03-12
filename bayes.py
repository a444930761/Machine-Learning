import numpy as np


def loadDataSet():#生成一个文本集
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #0代表正常，1代表敏感字
    return postingList,classVec

def createVocabList(dataSet):#利用文本集，生成一个词汇表
    vocabSet = set() #确保词汇表的唯一性
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setofwordsvec(vocabList,inputSet):#检查句中的单词是否在词汇表里存在
    #(词集模型，即只判断改词是否出现)
    returnvec = [0]*len(vocabList) #依据词汇表的长度生成一个全为0的向量
    for word in inputSet:
        if word in vocabList:
            returnvec[vocabList.index(word)] = 1 #如果单词存在词汇表，则将词汇表
            #对应的值设为1
        else:
            print('the word:{}is not in my vocabulary'.format(word))
    return returnvec

def bagofwordsvec(vocabList,inputSet):#检查句中的单词是否在词汇表里存在
    #(词袋模型，统计每个词出现的次数)
    returnvec = [0]*len(vocabList) #依据词汇表的长度生成一个全为0的向量
    for word in inputSet:
        if word in vocabList:
            returnvec[vocabList.index(word)] += 1 #如果单词存在词汇表，则将词汇表
            #对应的值加1
        else:
            print('the word:{}is not in my vocabulary'.format(word))
    return returnvec
#朴素贝叶斯分类器通常有两种实现方式，一种基于贝努利模型实现，一种基于多项式模型实现
#贝努利模型不考虑词出现的次数，只考虑词出不出现，相当于每个词的权重都是一样的
#多项式模型考虑词出现的次数，即给词赋予不一样的权重
    

listOposts,listclasses = loadDataSet()
myvocablist = createVocabList(listOposts)
#setofwordsvec(myvocablist)
'''初版    
def trainNBO(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #获取有多少条文本
    numWords = len(trainMatrix[0]) #获取文本长度
    pAbusive = np.sum(trainCategory)/float(numTrainDocs)#带有敏感字的文档和是3，
    #除以总文档数，即为是敏感文档的概率p(1)
    p0Num = np.zeros(numWords)#根据文本长度设定一个全0向量
    p1Num = np.zeros(numWords)#这里注意，生成的类型是np.ndarray
    p0Denom = 0
    p1Denom = 0
    for i in range(numTrainDocs):#遍历所有文本
        if trainCategory[i] == 1:#如果对应的标签是1，即敏感文本
            p1Num += trainMatrix[i] #统计文本中所有单词出现的次数
            #因为类型是np.ndarray，所以这里对应位置的值是直接相加的
            p1Denom += np.sum(trainMatrix[i]) #这里统计共有多少词
        else:#因为此例只有两个特征，即0或1，所以if后可直接else，否则要多加判断，
            #且要新增对应的统计变量
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom #这里计算敏感文本中，每个词占该类型下所有词的比例，即p(wi|c1)
    p0Vect = p0Num/p0Denom #这是p(wi|c0) #i和0是下标
    return p0Vect,p1Vect,pAbusive
    '''
#根据朴素贝叶斯假设，p(w|c) = p(w1|c)p(w2|c)...p(wn|c),因此我们要避免其中一项为0
#所以上述代码中，p0Num及p1Num的定义我们改为np.ones(numWords),同时将p0Denom和p1Denom初始化为2
#关于p0Vect和p1Vect的定义中，当因子非常小时，该变量值也小，那么p(w|c) = p(w1|c)p(w2|c)...p(wn|c)
#就很有可能下溢或者得不到正确答案，这里我们将其采用自然对数进行处理。改为：p1Vext = np.log(p1Num/p1Denom)
#f(x)和ln(f(x))在趋势上一致

def trainNBO(trainMatrix,trainCategory):
    '''计算各类文档以及各种词出现在各类文档的概率
    input：训练集数据trainMatrix及类型trainCategory
    return: 各类别概率
    '''
    numTrainDocs = len(trainMatrix) #获取有多少条文本
    numWords = len(trainMatrix[0]) #获取文本长度
    pAbusive = np.sum(trainCategory)/float(numTrainDocs)#带有敏感字的文档和是3，
    #除以总文档数，即为是敏感文档的概率p(1)
    p0Num = np.ones(numWords)#根据文本长度设定一个全0向量
    p1Num = np.ones(numWords)#这里注意，生成的类型是np.ndarray
    p0Denom = 0
    p1Denom = 0
    for i in range(numTrainDocs):#遍历所有文本
        if trainCategory[i] == 1:#如果对应的标签是1，即敏感文本
            p1Num += trainMatrix[i] #统计文本中所有单词出现的次数
            #因为类型是np.ndarray，所以这里对应位置的值是直接相加的
            p1Denom += np.sum(trainMatrix[i]) #这里统计共有多少词
        else:#因为此例只有两个特征，即0或1，所以if后可直接else，否则要多加判断，
            #且要新增对应的统计变量
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom) #这里计算敏感文本中，每个词占该类型下所有词的比例，即p(wi|c1)
    p0Vect = log(p0Num/p0Denom) #这是p(wi|c0) #i和0是下标
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):#计算最终概率并对比
    p1 = np.sum(vec2Classify*p1Vec) + np.log(pClass1)#因为转成log了，所以原定理中
    #相乘的部分通过相加实现，另外定理中还有一个分母，这里也没用，是因为要对比的分母是一样的
    #因此，这里只对比分子
    p0 = np.sum(vec2Classify*p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setofwordsvec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNBO(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setofwordsvec(myVocabList,testEntry))
    print(testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setofwordsvec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

'''使用朴素贝叶斯过滤垃圾邮件'''
import re


def textParse(bigString):
    listoftokens = re.split(r'\W*',bigString)#将字符串进行分割
    return [tok.lower() for tok in listoftokens if len(tok)>2] #将分割后的词汇全部转为小写，
    #并排除长度小于2的词
    
def spamTest():
    doclist = []
    classlist = []
    fulltext = []
    for i in range(1,26):#打开所有邮件样本，并汇总其中的文本及词汇
        emailText = open('D:/Anaconda/test/机器学习/Ch04/email/spam/{}.txt'.format(i),encoding='gbk').read()
        wordlist = textParse(emailText)
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(1)
        emailText = open('D:/Anaconda/test/机器学习/Ch04/email/ham/{}.txt'.format(i),encoding='gbk').read()
        wordlist = textParse(emailText)
        doclist.append(wordlist)
        fulltext.extend(wordlist)
        classlist.append(0)
    vocablist = createVocabList(doclist)#建立词汇集
    trainingSet = list(range(50))#总文档数是50
    testSet = []
    for i in range(10):#随机抽取10个文档加入到测试集
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:#剩下的40个文档为训练集
        trainMat.append(setofwordsvec(vocablist,doclist[docIndex]))
        #将剩下40个文档转化为向量后放入trainMat列表中
        trainClasses.append(classlist[docIndex])#将剩下40个文档的对应类型放到trainClasses列表中
    p0V,p1V,pSpam = trainNBO(np.array(trainMat),np.array(trainClasses))
    #计算概率
    errorCount = 0
    for docIndex in testSet:#利用测试集中的文档验证错误率
        wordVector = setofwordsvec(vocablist,doclist[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classlist[docIndex]:
            errorCount += 1
    print('the error rate is :',float(errorCount)/len(testSet))
    
#随机选择一部分作为训练集，二剩余部分作为测试集的过程称为留存交叉验证
    
'''对新闻进行分类
朴素贝叶斯主要是二分类，如果使用多（C）分类，可采用以下两种方法：
1. one-vs-one对于c类问题，将任意两类之间训练一个分类器，
这样一共就有c(c-1)/2个分类器。对于测试样本，用这c(c-1)/2个分类器进行预测，
再用投票法决定其类别。
2. one-vs-all (or one-vs-rest)这类方法训练c个分类器，
第i个分类器的功能是将第i类样本与其它所有类的样本区分开，即将第i类当作正类，
其它所有类样本当作负类，训练得到第i个分类器。对于测试样本，用这c个分类器进行预测，
其预测类别为函数值或者概率最大的那一类。（例如，每个NaiveBayes分类器对样本预测时，
都会得到属于第i类的概率，概率最大的那一类，即为预测类别）。

'''
path = 'D:/Anaconda/test/SogouC/Sample/'
import os
import jieba
def loadtxt(path):
    nowdir = os.getcwd()#获取当前工作路径
    os.chdir(path) #更改当前工作路径到文档位置
    datalist = [] #存储文档列表
    dataclass = [] #存储文档对应类别列表
    for i in os.listdir():
        filepath = path + i
        os.chdir(filepath)
        for j in os.listdir(): #遍历所有文件
            filepath2 = filepath + '/' + j
            with open(filepath2,'r',encoding='utf-8') as f:
                raw = f.read()
                datalist.append(list(jieba.cut(raw)))
                dataclass.append(i)
    os.chdir(nowdir)
    return datalist,dataclass

def trantest(data,label,p=0.2):
    '''
    input：要划分的数据集data；数据集对应的类别label，划分的比例p
    return：训练数据集traingdata及类别trainglabel、测试数据集testdata及类别testlabel，数据集allword
    '''
    num = int(len(data)*p)
    data = list(zip(data,label)) #将数据和对应的label拼装成元组
    np.random.shuffle(data) #打乱顺序
    traingdata = data[num:] #选出训练数据集
    traingdata,trainglabel = zip(*traingdata)
    testdata = data[:num] #选出测试数据集
    testdata,testlabel = zip(*testdata)
    
    worddict = {}
    for i in traingdata:
        for j in i:
            if j in worddict.keys():
                worddict[j] += 1
            else:
                worddict[j] = 1
                
    sortword = sorted(worddict.items(),key=lambda x:x[1],reverse=True)
    #排序后的格式是[(词，次数).()...()]
    allword,wordnum = zip(*sortword)#压缩后的形式上是[(词,...词),(次数,...次数)]
    allword = list(allword)#将元组转为列表
    
    return traingdata,trainglabel,testdata,testlabel,allword
    
def getdelword():
    delword = set()
    f = open('D:/data/Machine-Learning/Naive Bayes/stopwords_cn.txt','r',encoding='utf-8')
    for i in f.readlines():
        if len(i.strip()):
            delword.add(i.strip())
    f.close()
    return list(delword)

def makewordset(word,delword,n=100):
    '''
    input：数据集word，要删除的数据集delword，要删除词频排名前n
    return: 整理后的数据集featur_word
    '''
    feature_word = []
    m = 0
    for i in range(n,len(word)):
        if m >1000:
            break
        if word[i] not in delword and 1<len(word[i])<5 and not word[i].isdigit():
            feature_word.append(word[i])          
        m += 1 
    return feature_word

def setofwordsvec(feauter_word,traingdata):
    '''将文本集转化为向量
    input: 词库feauter_word,训练文本集：traingdata
    return： 向量文本集traingvec
    '''
    m = len(feauter_word)   
    traingvec = []
    for i in traingdata:
        n = np.ones(m)
        for j in i:
            if j in feauter_word:
                n[feauter_word.index(j)] += 1
        traingvec.append(n)
    return traingvec

def train(feauter_word,traingvec,datalabel):
    '''计算概率
    input:词库feauter_word，训练集traingvec及类别datalabel
    return：各概率
    '''
    labeldict = {}
    for i in datalabel:
        if i in labeldict.keys():
            labeldict[i] += 1
        else:
            labeldict[i] = 1
    labelrator = {}
    for i in labeldict.keys():
        labeldict[i] = labeldict[i]/len(datalabel) #计算每个分类的概率
        n = np.zeros(len(feauter_word))
        m = 2
        for j in datalabel:
            if j == i:
                n += traingvec[datalabel.index(j)]
                m += np.sum(traingvec[datalabel.index(j)])
        labelrator[i] = np.log(n/m)
    
    return labeldict,labelrator 

def returnresult(testdata,labeldict,labelrator):
    a = {}
    for i in labeldict.keys():
        p = np.sum(testdata*labelrator[i]) + np.log(labeldict[i])
        a[i] = p
    sorta = sorted(a.items(),key=lambda x:x[1])
    return sorta[-1]
        

if __name__=='__main__':
    datalist,datalabel =  loadtxt(path)
    traingdata,trainglabel,testdata,testlabel,allword = trantest(datalist,datalabel,p=0.2)
    delword = getdelword()
    feauter_word = makewordset(allword,delword)
    traingvec = setofwordsvec(feauter_word,traingdata)
    testvec = setofwordsvec(feauter_word,testdata)
    labeldict,labelrator = train(feauter_word,traingvec,trainglabel)
    m=0
    rightcount = 0
    for i in testvec: 
        result = returnresult(np.array(i),labeldict,labelrator)
        if result[0] == testlabel[m]:
            rightcount += 1 
        m += 1
    print(rightcount/m)
        
'''使用sklearn实现朴素贝叶斯
在scikit-learn中，一共有3个朴素贝叶斯的分类算法类。分别是GaussianNB，MultinomialNB
和BernoulliNB。其中GaussianNB就是先验为高斯分布的朴素贝叶斯，MultinomialNB就是
先验为多项式分布的朴素贝叶斯，而BernoulliNB就是先验为伯努利分布的朴素贝叶斯
class sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
'''    
from sklearn.naive_bayes import MultinomialNB

def datavec(traingdata,testdata,feauter_word): #将数据转化为向量
    def feautervec(data,feauter_word):
            words = set(data)
            feautervec = [1 if i in words else 0 for i in feauter_word]
            return feautervec
            #这里设置默认为0，是因为MultinomialNB的alpha参数会使用平滑解决概率为0的问题
    traingvec = [feautervec(data,feauter_word) for data in traingdata]
    testvec = [feautervec(data,feauter_word) for data in testdata]
    return traingvec,testvec

def TextClassifier(traingdata,testdata,trainglabel,testlabel):
    classifier = MultinomialNB().fit(traingdata, trainglabel)
    test_accuracy = classifier.score(testdata, testlabel)
    return test_accuracy

if __name__=='__main__':
    datalist,datalabel =  loadtxt(path)
    traingdata,trainglabel,testdata,testlabel,allword = trantest(datalist,datalabel,p=0.2)
    delword = getdelword()
    feauter_word = makewordset(allword,delword)
    traingvec,testvec = datavec(traingdata,testdata,feauter_word)
    
    test_accuracy_list = []
    test_accuracy = TextClassifier(traingvec, testvec, trainglabel, testlabel)
    test_accuracy_list.append(test_accuracy)
    ave = lambda c: sum(c) / len(c)
 
    print(ave(test_accuracy_list))
    
    

        