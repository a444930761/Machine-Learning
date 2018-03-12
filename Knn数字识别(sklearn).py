import numpy as np
import os

traingfile = '机器学习实战/Ch02/digits/trainingDigits/'
testfile = '机器学习实战/Ch02/digits/testDigits/'

def loadfile(file):
    '''读取文件函数
    Parameters：file - 要读取的文件
    Returns：dataSet - 读取后的数据集
             labels - 数据集对应的标签
    Author：Li Wei
    '''
    dataSet = []
    labels = []
    for i in os.listdir(file):
        labels.append(i.split('_')[0])
        filename = file + i
        f = open(filename)
        data = ''
        for ii in f.readlines():
            data += ii.strip()
        dataSet.append(list(data))
    return np.array(dataSet).astype('int'),labels

def classify(inX,dataSet,labels,k):
    '''分类函数
    Parameters：inX - 要分类的数据
                dataSet - 数据集
                labels - 数据集对应的标签
    Returns： 要分类数据的标签
    Author：Li Wei
    '''
    m = dataSet.shape[0]
    diffMat = (np.tile(inX,(m,1)) - dataSet) ** 2
    l = (diffMat.sum(axis=1)) ** 0.5
    lsort = l.argsort()
    ldict = {}
    for i in range(k):
        label = labels[lsort[i]]
        ldict[label] = ldict.get(label,0) + 1
    ldict = sorted(ldict.items(),key=lambda x:x[1],reverse=True)
    return ldict[0][0]

def testclassify():
    '''验证函数
    Parameters：无
    Returns：错误率
    Author：Li Wei
    '''
    traingdata,trainglabel = loadfile(traingfile)
    testdata,testlabel = loadfile(testfile)
    
    m = testdata.shape[0]
    errcount = 0
    for i in range(m):
        result = classify(testdata[i,:],traingdata,trainglabel,4)
        print('预测结果是{}，实际结果是{}'.format(result,testlabel[i]))
        if result != testlabel[i]:
            errcount += 1
    print('错误率为{:%}.'.format(errcount/m))

if __name__ == '__main__':
    testclassify()
    
    
'''使用python的sklearn库实现KNN算法'''
from sklearn.neighbors import KNeighborsClassifier as KNN

def testclassify2():
    '''KNN模型函数
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    traingdata,trainglabel = loadfile(traingfile)
    testdata,testlabel = loadfile(testfile)
    
    m = testdata.shape[0]
    errcount = 0
    neigh = KNN(n_neighbors=5) #构建KNN模型
    neigh.fit(traingdata,trainglabel) #对模型进行训练
    for i in range(m):
        result = neigh.predict(testdata[i,:].reshape(1,-1)) #利用模型进行输出 
        #这里需要注意，predict参数X的shape为[n_samples,n_features]，例如（1，1024）
        #而testdata[i,:]为（1024，），所以用reshape更改下
        print('预测结果是{}，实际结果是{}'.format(result,testlabel[i]))
        if result != testlabel[i]:
            errcount += 1
    print('错误率为{:%}.'.format(errcount/m))