# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:52:41 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
def loadfile():
    '''读取文档函数
    Parameters：file - 要读取的文档
    Returns：xcord - x轴数据
             ycord - y轴数据
    Author：Li Wei
    '''
    file = 'Ch08/ex0.txt'
    f = open(file,encoding='utf-8')
    dataArr = []
    for i in f.readlines():
        dataArr.append(i.strip().split('\t'))
    data_np = np.array(dataArr).astype(float)
    xcord = data_np[:,:-1]
    ycord = data_np[:,-1]
    f.close()
    return xcord,ycord

def plotDataSet():
    '''绘制数据点图形
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    x,y = loadfile()
    x = x[:,-1]
    plt.scatter(x,y,alpha = 0.5,s = 20)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
    
#plotDataSet()
    
def standRegres(xArr,yArr):
    '''计算回归系数w
    Parameters：xArr - x数据集
                yArr - y数据集
    Returns：w - 回归系数
    Author：Li Wei
    公式：w = (X.T * X).I * X.T * Y
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T #这里注意y要进行转置
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0:
        print('矩阵为奇异矩阵，不能求逆')
        return 
    w = xTx.I * (xMat.T * yMat)
    return w

def plotRegression():
    '''绘制回归曲线和数据点
    Parameters:无
    Returns： 无
    Author：Li Wei
    '''
    xArr,yArr = loadfile()
    w = standRegres(xArr,yArr)
    xMat = np.mat(xArr)#; yMat = np.mat(yArr)
    xCopy = xMat.copy()
    xCopy.sort(0) #因为要画回归线,所以从小到大排序
    yHat = xCopy * w
    flg = plt.figure()
    ax = flg.add_subplot(111)
    ax.plot(xCopy[:,-1],yHat,c='red')
    ax.scatter(xArr[:,-1],yArr,s=20,c='blue',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()

if __name__=='__main__':
    plotRegression()

'''
利用corrcoef查看预测值和实际值的线性相关性
'''    
xArr,yArr = loadfile()
w = standRegres(xArr,yArr)
xMat = np.mat(xArr); yMat = np.mat(yArr)
yHat = xMat * w
print(np.corrcoef(yHat.T,yMat))

'''局部加权线性回归'''
def lwlr(testPoint,xArr,yArr,k = 1):
    '''使用局部加权线性回归计算回归系数w
    Parameters: testPoint - 测试样本点
                xArr - x数据集
                yArr - y数据集
                k - 高斯核的k，自定义系数
    Returns：w - 回归系数
    Author：Li Wei
    高斯核函数公式：w(i,i) = exp((||xi-x||)**2/(-2*k**2))
    局部权重回归公式：w = (xMat.T * (weights * xMat)).I * (xMat.T * (weights * yMat))
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye((m))) #创建权重对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat * diffMat.T/(-2 * k**2)) #利用高斯核函数求权重
        #公式：w(i,i) = exp((||xi-x||)**2/(-2*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0:
        print('矩阵为奇异矩阵，不能求逆')
        return
    w = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * w

def lwlrTest(testArr,xArr,yArr,k=1):
    '''局部加权线性回归测试
    Parameters：testArr - 测试数据集
                xArr - x数据集
                yArr - y数据集
                k - 高斯核的k，自定义参数
    Returns：w - 回归系数
    Author：Li Wei
    '''
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def plotlwlrRegression():
    '''绘制多条局部加权回归曲线
    Parameters：无
    Returns： 无
    Author：Li Wei
    '''
    font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=14)
    xArr,yArr = loadfile()
    yHat_1 = lwlrTest(xArr,xArr,yArr,1)
    yHat_2 = lwlrTest(xArr,xArr,yArr,0.01)
    yHat_3 = lwlrTest(xArr,xArr,yArr,0.003)
    xMat = np.mat(xArr)#; yMat = np.mat(yArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig,axs = plt.subplots(nrows=3,ncols=1,sharex=False,sharey=False,figsize=(10,8))
    axs[0].plot(xSort[:,1],yHat_1[srtInd],c='red')
    axs[1].plot(xSort[:,1],yHat_2[srtInd],c='red')
    axs[2].plot(xSort[:,1],yHat_3[srtInd],c='red')
    axs[0].scatter(xArr[:,1],yArr,s=20,c='blue',alpha=0.5)
    axs[1].scatter(xArr[:,1],yArr,s=20,c='blue',alpha=0.5)
    axs[2].scatter(xArr[:,1],yArr,s=20,c='blue',alpha=0.5)
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0',FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01',FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003',FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')  
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')  
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')  
    plt.xlabel('X')
    plt.show()
if __name__=='__main__':   
    plotlwlrRegression()
    
'''预测鲍鱼年龄'''
def loadfile2():
    '''读取文档函数
    Parameters：file - 要读取的文档
    Returns：xcord - x轴数据
             ycord - y轴数据
    Author：Li Wei
    '''
    file = 'Ch08/abalone.txt'
    f = open(file,encoding='utf-8')
    dataArr = []
    for i in f.readlines():
        dataArr.append(i.strip().split('\t'))
    data_np = np.array(dataArr).astype(float)
    xcord = data_np[:,:-1]
    ycord = data_np[:,-1]
    f.close()
    return xcord,ycord

def rssError(yArr,yHatArr):
    '''误差大小评价函数
    Parameters：yArr - 真实数据
                yHatArr - 预测数据
    Returns：误差大小
    Author：Li Wei
    '''
    return ((yArr - yHatArr)**2).sum()

xArr,yArr = loadfile2()
print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
yHat01 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.1)
yHat1 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
yHat10 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)
print('k=0.1时,误差大小为:',rssError(yArr[0:99], yHat01.T))
print('k=1  时,误差大小为:',rssError(yArr[0:99], yHat1.T))
print('k=10 时,误差大小为:',rssError(yArr[0:99], yHat10.T))
print('')
print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
yHat01 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 0.1)
yHat1 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 1)
yHat10 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 10)
print('k=0.1时,误差大小为:',rssError(yArr[100:199], yHat01.T))
print('k=1  时,误差大小为:',rssError(yArr[100:199], yHat1.T))
print('k=10 时,误差大小为:',rssError(yArr[100:199], yHat10.T))
print('')
print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
print('k=1时,误差大小为:', rssError(yArr[100:199], yHat1.T))
ws = standRegres(xArr[0:99], yArr[0:99])
yHat = np.mat(xArr[100:199]) * ws
print('简单的线性回归误差大小:', rssError(yArr[100:199], yHat.T.A))


'''岭回归
最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差
从而得到更好的估计
'''

def ridgeRegres(xMat,yMat,lam=0.2):
    '''岭回归函数
    Parameters：xMat - x数据集
                yMat - y数据集
                lam - 缩减系数
    Returns：w - 回归系数
    Author：Li Wei
    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(denom) == 0:
        print('奇异矩阵不能转置')
        return
    w = denom.I * (xMat.T * yMat)
    return w

def ridgeTest(xArr,yArr):
    '''岭回归测试
    Parameters： xArr - x数据集
                 yArr - y数据集
    Returns： wMat - 回归系数矩阵
    Author：Li Wei
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat,axis=0)
    yMat = yMat - yMean
    xMean = np.mean(xMat,axis=0)
    xVar = np.var(xMat,axis=0)
    xMat = (xMat - xMean) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts,xMat.shape[1]))
    for i in range(numTestPts):
        w = ridgeRegres(xMat,yMat,np.exp(i - 10))#将lam以指数级发生变化，能更好的观测到lam值从小到大的影响
        wMat[i,:] = w.T
    return wMat

def plotMat():
    '''绘制岭回归系数矩阵
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    abX,abY = loadfile2()
    redgeWeights = ridgeTest(abX,abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title('log(lambada)与回归系数的关系',FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()

'''plotMat() #绘制回归系数和lambda的关系
从图中可以看到，当lambda为0时，所有系数和之前的线性回归一直，
当lambada最大，回归系数开始缩减到0，在中间存在着lambda的最优值'''

'''前向逐步回归'''

def rssError(yArr,yHatArr):
    '''计算平方误差
    Parameters：yArr - 真实值
                yHatArr - 预测值
    Returns：平方误差
    Author：Li Wei
    '''
    return ((yArr - yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt = 100):
    '''向前逐步线性函数
    Parameters：xArr - x输入数据
                yArr - y预测数据
                eps - 每次迭代需要调整的步长
                numIt - 迭代次数
    Returns：returnMat - numIt次迭代的回归系数矩阵
    Author：Li Wei
    '''
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xMat = (xMat - xMat.mean(axis=0)) / xMat.var(axis=0) #数据标准化
    yMat = yMat - yMat.mean()
    m,n = xMat.shape
    returnMat = np.zeros((numIt,n)) #初始化回归系数矩阵
    ws = np.zeros((n,1)) #初始化权重
    wsMax = ws.copy()
    for i in range(numIt):
        minerror = float('inf') #初始化最小误差为无穷大
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign #迭代更改每一列的权重
                yTest = xMat * wsTest #计算预测值
                rssE = rssError(yMat.A,yTest.A) #计算平方误差
                if rssE < minerror:
                    minerror = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def plotstageWiseMat():
    '''绘制岭回归系数矩阵
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    xArr,yArr = loadfile2()
    returnMat = stageWise(xArr,yArr,0.005,1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title('前向逐步回归:迭代次数与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel('迭代次数', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel('回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 15, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()
#plotstageWiseMat() #绘制迭代次数与回归系数图
    #从图中可以看到，有些特征的系数一直约为0，即，不对目标造成影响，这类特征可以不考虑
    
    
'''预测乐高价格'''
from bs4 import BeautifulSoup
 
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
    函数说明:从页面读取数据，生成retX和retY列表
    Parameters:
        retX - 数据X
        retY - 数据Y
        inFile - HTML文件
        yr - 年份
        numPce - 乐高部件数目
        origPrc - 原价
    Returns:
        无
    Author：Li Wei
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r = "%d" % i)
    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r = "%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r = "%d" % i)
         
def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
        无
    Author：Li Wei
    """
    scrapePage(retX, retY, 'Ch08/setHtml/lego8288.html', 2006, 800, 49.99)                #2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, 'Ch08/setHtml/lego10030.html', 2002, 3096, 269.99)                #2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, 'Ch08/setHtml/lego10179.html', 2007, 5195, 499.99)                #2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, 'Ch08/setHtml/lego10181.html', 2007, 3428, 199.99)                #2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, 'Ch08/setHtml/lego10189.html', 2008, 5922, 299.99)                #2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, 'Ch08/setHtml/lego10196.html', 2009, 3263, 249.99)                #2009年的乐高10196,部件数目3263,原价249.99
 
if __name__ == '__main__':
    lgX = [] #标准价格
    lgY = [] #出售价格
    setDataCollect(lgX, lgY)
    
def getdata(lgX,lgY):
    '''获取数据
    Parameters：lgX - 初始特征集
                lgY - 真实标签
    Returns：xArr - x数据集
             yArr - y数据集
    Author：Li Wei
    '''
    xArr = []
    yArr = []
    for i in range(len(lgX)):
        xArr.append([1] + lgX[i])
        yArr.append(lgY[i])
    return xArr,yArr

def standRegres():
    '''使用简单线性回归求系数
    Parameters：无
    Returns：w - 系数
    Author：Li Wei
    '''
    xArr,yArr = getdata(lgX,lgY)
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = (xMat.T * xMat)
    if np.linalg.det(xTx) == 0:
        print('奇异矩阵，无法求逆')
        return 
    w = xTx.I * (xMat.T * yMat)
    return w
'''
w = standRegres()
print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (w[0],w[1],w[2],w[3],w[4]))
以上是简单的线性回归，虽然这个模型对于数据拟合得很好
但是套件里的部件数量越多，售价反而降低了，这是不合理的。
下面使用岭回归来进行交叉验证
'''

def regularize(xMat,yMat):
    '''标准化函数
    Parameters：xMat - x数据集
                yMat - y数据集
    Returns：inxMat - 标准化后的x数据集
             inyMat - 标准化后的y数据集
    Author：Li Wei
    '''
    inxMat = (xMat - xMat.mean(axis=0)) / xMat.var(axis=0)
    inyMat = yMat - yMat.mean()
    return inxMat,inyMat
   
def rssError(yArr,yHatArr):
    '''计算平方误差函数
    Parameters：yArr - 真实值
                yHATArr - 预测值
    Returns：平方误差
    Author：Li Wei
    '''
    return ((yArr - yHatArr)**2).sum()

def corssValidation(xArr,yArr,numVal = 10):
    '''交叉验证岭回归
    Parameters：xArr - x数据集
                yArr - y数据集
                numVal - 交叉验证次数
    Returns：wMat - 回归系数矩阵
    Author：Li Wei
    '''
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal,30))
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        np.random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)
        for k in range(30):
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX,0)
            varTrain = np.var(matTrainX,0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A,np.array(testY))
    meanErrors = errorMat.mean(axis = 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    #np.nonzero以元组的形式输出对象的非零元素的索引
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights / varX
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))

#corssValidation(lgX,lgY)

'''使用sklearn的linear_model'''
from sklearn import linear_model
reg = linear_model.Ridge(alpha=0.5)
reg.fit(lgX,lgY)
print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))