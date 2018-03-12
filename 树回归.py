# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def testCart(data,num,thresh):
    '''展示阈值划分
    Parameters：data - 要划分的数据集
                num - 要划分的特征
                thresh - 划分设定的阈值
    Returns：m0 - 划分后的数据集
             m1 - 划分后的数据集
    Author：Li Wei
    '''
    m0 = data[data[:,num] < thresh]
    m1 = data[data[:,num] >= thresh]
    return m0,m1

data = np.eye((5))
m0,m1 = testCart(data,1,0.5)
print('原始数据集:',data)
print('划分后的数据集m0：',m0)
print('划分后的数据集m1：',m1)

def loadfile():
    '''读取数据函数
    Parameters：无
    Returns：data_np - 读取后的数据
    Author：Li Wei
    '''
    file = 'Ch09/ex00.txt'
    f = open(file)
    data = []
    for i in f.readlines():
        data.append(i.strip().split('\t'))
    data_np = np.array(data).astype(float)
    return data_np
data2 = loadfile()
'''绘制数据图'''
plt.scatter(data2[:,0],data2[:,1])
plt.title('DataSet')
plt.xlabel('X')
plt.show()

'''
回归树(regression tree) 每个叶节点包含单个值
模型树(model tree) 每个叶节点包含一个线性方程
'''
def loadDataSet(filename):
    '''加载数据
    Parameters：filename - 数据文件
    Returns：dataMat - 加载后的数据列表
    Author：Li Wei
    '''
    dataMat = []
    f = open(filename)
    for line in f.readlines():
        curline = line.strip().split('\t')
        fltLine = list(map(float,curline)) #将数据转化为浮点型
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    '''切割函数
    Parameters：dataSet - 要切割的数据集
                feature - 待切割的特征
                value - 该特征的切割值
    Returns：m0 - 大于切割值的数据集合
             m0 - 小于等于切割值的数据集合
    Author：Li Wei
    '''
    m0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    m1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return m0,m1

def regLeaf(dataSet):
    '''建立叶节点函数，目标(y)的均值,chooseBestSplit()函数不再对数据进行切分时，调用
    该函数获取叶节点的模型
    Parameters：dataSet - 数据集
    Returns：y的均值
    Author：Li Wei
    '''
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    '''误差计算函数，目标(y)的总方差(均方差乘以样本个数)
    Parameters：数据集
    Returns：误差
    Author：Li Wei
    '''
    return np.var(dataSet[:,-1]) * dataSet.shape[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    ''' 选择最优切分点
    Parameters：dataSet - 要切分的数据集
                leafType - 生成叶节点函数
                errType - 误差计算函数
                ops - 预剪枝参数
    Returns： bestIndex - 最优切分的特征索引
              bestValue - 最优切分的值
    Author：Li Wei
    '''
    tolS = ops[0] #设定误差下降值
    tolN = ops[1] #设定切分最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #如果目标(y)所有值相等,则退出
        return None, leafType(dataSet)
    m,n = dataSet.shape
    S = errType(dataSet)
    bestS = float('inf'); bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): #循环所有特征
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]): #循环每个特征的所有值
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            #如果切分后样本数小于设定的标准，则退出此次循环
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN): 
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果最优的切分后的误差降低小于设定的标准，则不进行切分
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    #如果最优切分后的样本数小于设定的标准，则不进行切分
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    '''创建树函数
    Parameters：dataSet - 数据集
                leafType - 建立叶节点的函数 
                errType - 误差计算函数
                ops - 预剪枝参数
    Returns: retTree - 构建好的树
    Author：Li Wei
    '''
    feat, val = chooseBestSplit(dataSet,leafType,errType,ops)
    #满足停止条件，返回None和模型的值
    if feat == None: 
        return val   #如果是回归树，val是一个值，如果是模型树，val是一个线性方程
    retTree = {} 
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #不满足停止条件，则将数据分成两部分，递归调用createTree函数
    lSet,rSet = binSplitDataSet(dataSet,feat,val) 
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


#myDat = loadfile()
#myMat = np.mat(myDat)
#createTree(myMat)

'''利用多分类数据来进行回归树构建'''
file2 = 'Ch09/ex0.txt'
def plotData(file):
    '''绘制数据集函数
    Parameters：file - 数据文件
    Returns：无
    Author：Li Wei
    '''
    x = []
    y = []
    f = open(file)
    for i in f.readlines():
        line = i.strip().split('\t')
        x.append(line[1])
        y.append(line[2])
    plt.scatter(x,y)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
#plotData(file2) #可以看到数据分为5类    

#myDat = loadDataSet(file2)
#myMat = np.mat(myDat)
#createTree(myMat) #可以看到结果有5个叶节点
    
#file3 = 'D:/Anaconda/test/机器学习/Ch09/ex2.txt'
#myDat = loadDataSet(file3)
#myMat2 = np.mat(myDat)
#myTree = createTree(myMat2,ops=(0,1)) #tolS对误差的数量级非常敏感
    
'''后剪枝技术'''
def isTree(obj):
    '''判断是否有节点函数(字典)
    Parameters：obj - 要判断的对象
    Returns：True or False
    Author：Li Wei
    '''
    return (type(obj) == dict)

def getMean(tree):
    '''对数的两个节点进行塌陷处理，即计算树两个节点的平均值
    Parameters：tree - 要递归计算的树
    Retunrs：树两个节点的平均值
    Author：Li Wei
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2
    
def prune(tree,testData):
    '''后剪枝函数
    Parameters：tree - 待剪枝的树
                testData - 剪枝所需的测试数据
    Returns：tree - 剪枝后的树
    Author：Li Wei
    '''
    if testData.shape[0] == 0: #测试集如果为空，直接返回平均值
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    #如果子集为树，则递归进行剪枝操作    
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
        
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    #子集不为树后，开始计算合并后的误差    
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        #计算合并前的误差
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) +\
                        np.sum(np.power(rSet[:-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right']) / 2
        #计算合并后的误差
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        #如果合并后的误差小于合并前，则合并，否则不合并直接返回
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree
    
'''
对上面生成的myTree进行后剪枝操作
file4 = 'Ch09/ex2test.txt'    
myDatTest = loadDataSet(file4)
myMat2Test = np.mat(myDatTest)
prune(myTree,myMat2Test)   #可以看到部分节点被合并，但效果依然不理想，一般地，为了寻求
#最佳模型，可同时使用后剪枝和预剪枝
'''


'''构建模型树'''
def linearSolve(dataSet):
    '''求解简单的线性回归系数函数
    Parameters：dataSet - 要回归的数据
    Returns：w - 回归系数
             X - x数据集
             Y - y数据集
    Author：Li Wei
    '''
    m,n = dataSet.shape
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,:n-1] #X的第一列是x0，默认值都是1
    Y = dataSet[:,-1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        print('奇异矩阵无法求逆')
        return
    w = xTx.I * (X.T * Y)
    return w,X,Y

def modelLeaf(dataSet):
    '''生成叶节点函数，叶节点为线性的系数
    Parameters：dataSet - 数据集
    Returns：w - 回归系数
    Author：Li Wei
    '''
    w,X,Y = linearSolve(dataSet)
    return w

def modelError(dataSet):
    '''计算模型误差函数
    Parameters：dataSet - 数据集
    Returns：误差值
    Author：Li Wei
    '''
    w,X,Y = linearSolve(dataSet)
    yHat = X * w
    return np.sum(np.power(yHat - Y,2))

'''构建模型树
file5 = 'Ch09/exp2.txt'
myDat = loadDataSet(file5)
myDat_np = np.array(myDat)
x = myDat_np[:,0]; y = myDat_np[:,-1]
plt.scatter(x,y)
myMat2 = np.mat(myDat)
myTree = createTree(myMat2,modelLeaf,modelError,ops=(1,10)) 
'''

def regTreeEval(model,inDat):
    '''回归树预测函数
    Parameters：model - 树的叶节点值(单个的值)
                inDat - 在回归树预测里没有实际意义，只是为了和下面的模型树预测函数保持一致
    Returns：model - 树的叶节点转化为浮点型作为预测值
    Author：Li Wei
    '''
    return float(model)

def modelTreeEval(model,inDat):
    '''模型树预测函数
    Parameters：model - 树的也节点值(回归系数)
                inDat - 进行模型预测的自变量
    Returns：模型预测后的值
    Author：Li Wei
    '''
    n = inDat.shape[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X * model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    '''
    Parameters：tree - 预测的参照树
                inData - 要预测的自变量
                modelEval - 计算预测结果的函数
    Returns: 返回预测结果
    Author：Li Wei
    '''
    if not isTree(tree):
        return modelTreeEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    '''通过参照树对测试数据进行预测
    Parameters：tree - 参照树
                testData - 要预测的测试数据
                modelEval - 求预测值的模型
    Returns：预测结果
    Author：Li Wei
    '''
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,np.mat(testData[i]),modelEval)
    return yHat

file6 = 'Ch09/bikeSpeedVsIq_train.txt'
file7 = 'Ch09/bikeSpeedVsIq_test.txt'
trainDat = loadDataSet(file6)
trainDat_np = np.array(trainDat)
x = trainDat_np[:,0]; y = trainDat_np[:,-1]
plt.scatter(x,y)
trainMat = np.mat(trainDat)
testMat = np.mat(loadDataSet(file7))
'''利用回归树进行预测'''
myTree = createTree(trainMat,ops=(1,20)) 
yHat = createForeCast(myTree,testMat[:,0])
np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]

'''利用模型树进行预测'''
myTree = createTree(trainMat,modelLeaf,modelError,(1,20))
yHat = createForeCast(myTree,testMat[:,0],modelTreeEval)
np.corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]


'''利用GUI对回归树调优'''
import tkinter

#gui小示例
root = tkinter.Tk()
myLabel = tkinter.Label(root,text='测试') #建立一个tkinter标签(label)
myLabel.grid() #将标签的位置告诉布局管理器，grid方法户将标签安排在一个二维的表格中，
#默认放置在0行0列
root.mainloop()


#构建树管理器界面的tkinter小部件
def reDraw(tolS,tolN):
    pass

def drawNewTree():
    pass

root = tkinter.Tk()

tkinter.Label(root,text='Plot Place Holder').grid(row=0,columnspan=3) #位置在0行，跨列度为3

tkinter.Label(root,text='tolN').grid(row=1,column=0) #添加label，位置在1行，0列
tolNentry = tkinter.Entry(root) #Entry为单行文本输入框 
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10') #输入框的默认值为10

tkinter.Label(root,text='tolS').grid(row=2,column=0)
tolSentry = tkinter.Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1')

tkinter.Button(root,text='ReDraw',command=drawNewTree).grid(row=1,column=3,rowspan=3)
#添加Button(按钮)，点击按钮，调用drawNewTree函数
chkBtnVar = tkinter.IntVar()
chkBtnVar = tkinter.Checkbutton(root,text='Model Tree',variable=chkBtnVar) #添加复选框
chkBtnVar.grid(row=3,column=0,columnspan=2)

reDraw.rawDat = np.mat(loadDataSet('Ch09/sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1,10)

root.mainloop()


'''在Tk的GUI上放置一个画布'''
root = tkinter.Tk()
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
reDraw.f = Figure(figsize=(5,4),dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
root.mainloop()

'''更改以上代码，将Matplotlib和Tkinter的代码集合'''
import matplotlib
matplotlib.use('TkAgg') #设定matplotlib的后端为TkAgg
import tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def reDraw(tolS,tolN):
    '''绘图函数
    Parameters：tolS - 切分后误差降低最低标准
                tolN - 切分后最少的样本数
    Returns：无
    Author：Li Wei
    '''
    reDraw.f.clf() #清空所有图像，防止新作的图和原来的图重叠
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get(): #检查复选框是否被选中
        if tolN < 2 :
            tolN = 2
        myTree = createTree(reDraw.rawDat,modelLeaf,modelError,(tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testData,modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testData)
    reDraw.a.scatter(reDraw.rawDat[:,0].T.tolist()[0],reDraw.rawDat[:,1].T.tolist()[0],s=5)
    reDraw.a.plot(reDraw.testData,yHat,linewidth=2)
    reDraw.canvas.show()
    
def getInputs():
    '''获取用户输入的tolN，tolS参数
    Parameters：无
    Returns：tolN - 切分后的最小样本数
             tolS - 切分后的误差降低最低标准
    Author：Li Wei
    '''
    try: tolN = int(tolNentry.get())
    except: #如果不能解析
        tolN = 10 #设置tolN为默认值
        print('请输入tolN的整数值') #给用户提示
        tolNentry.delete(0,END) #删除框里所有数据
        tolNentry.insert(0,'10')
    try: tolS = float(tolSentry.get())
    except:
        tolS = 1
        print('输入tolS的浮点数值')
        tolNentry.delete(0,END)
        tolNentry.insert(0,'1')
    return tolN,tolS

def drawNewTree():
    '''设置参数并生成新的图
    Parameters：无
    Returns：无
    Author：Li Wei
    '''
    tolN,tolS = getInputs() #调用getInputs函数获取参数值
    reDraw(tolS,tolN) #使用参数值绘制新的图形
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg') #设定matplotlib的后端为TkAgg
    import tkinter
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    root = tkinter.Tk()
    
    reDraw.f = Figure(figsize=(5,4),dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
    
#    tkinter.Label(root,text='Plot Place Holder').grid(row=0,columnspan=3) #位置在0行，跨列度为3
    
    tkinter.Label(root,text='tolN').grid(row=1,column=0) #添加label，位置在1行，0列
    tolNentry = tkinter.Entry(root) #Entry为单行文本输入框 
    tolNentry.grid(row=1,column=1)
    tolNentry.insert(0,'10') #输入框的默认值为10
    
    tkinter.Label(root,text='tolS').grid(row=2,column=0)
    tolSentry = tkinter.Entry(root)
    tolSentry.grid(row=2,column=1)
    tolSentry.insert(0,'1')
    
    tkinter.Button(root,text='ReDraw',command=drawNewTree).grid(row=1,column=2,rowspan=3)
    #添加Button(按钮)，点击按钮，调用drawNewTree函数
    chkBtnVar = tkinter.IntVar()
    chkBtn = tkinter.Checkbutton(root,text='Model Tree',variable=chkBtnVar) #添加复选框
    chkBtn.grid(row=3,column=0,columnspan=2)
    
    reDraw.rawDat = np.mat(loadDataSet('D:/Anaconda/test/机器学习/Ch09/sine.txt'))
    reDraw.testData = np.arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
    reDraw(1,10)
    
    root.mainloop()    