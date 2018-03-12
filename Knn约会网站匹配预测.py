import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines

def createDataSet():
    '''创建数据集函数
    Parameters：无
    Returns：group - 数据集
             labels - 数据对应的标签
    Author：Li Wei
    '''
    group = np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = list('AABB')
    return group,labels
        
def classify(inX,dataSet,labels,k):
    '''分类函数
    Parameters：inX - 要确定标签的数据
                dataSet - 数据集
                labels - 数据集对应的标签
                k - 决定标签的前几个数据
    Returns：数据的标签
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
            
group,labels = createDataSet()
'''
plt.scatter(group[:,0],group[:,1])
plt.xticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])
for i in range(group.shape[0]):
    plt.text(group[i,0]+0.02,group[i,1],labels[i])
'''
classify([0,0],group,labels,3)

def showdata(dataSet,labels):
    '''绘图函数
    Parameters：dataSet - 数据集
                labels - 数据集对应的标签
    Returns：无
    Author：Li Wei
    '''
    #设置汉字字体
    font = FontProperties(fname='c:/windows/fonts/msyhl.ttc', size=14)
    
    #给对应的标签设置不同的颜色
    labelcolors = []
    for i in labels:
        if i == '1':
            labelcolors.append('black')
        elif i == '2':
            labelcolors.append('orange')
        elif i == '3':
            labelcolors.append('red')

    fig,axs = plt.subplots(2,2,figsize=(13,8))
    #画出散点图,dataSet第一列(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.8
    axs[0][0].scatter(dataSet[:,0],dataSet[:,1],color=labelcolors,s=15,alpha=0.8)
    #设置标题，x轴label，y轴label
    axs_title = axs[0][0].set_title('每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs_xlabel = axs[0][0].set_xlabel('每年获得的飞行常客里程数',FontProperties=font)
    axs_ylabel = axs[0][0].set_ylabel('玩视频游戏所消耗时间占',FontProperties=font)
    #调整标题、x轴label、y轴label的属性
    plt.setp(axs_title,size=9,color='red')
    plt.setp(axs_xlabel,size=9,color='black')
    plt.setp(axs_ylabel,size=9,color='black')
    
    #画出散点图,dataSet第一列(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.8
    axs[0][1].scatter(dataSet[:,0],dataSet[:,2],color=labelcolors,s=15,alpha=0.8)
    axs_title = axs[0][1].set_title('每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs_xlabel = axs[0][1].set_xlabel('每年获得的飞行常客里程数',FontProperties=font)
    axs_ylabel = axs[0][1].set_ylabel('每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs_title,size=9,color='red')
    plt.setp(axs_xlabel,size=9,color='black')
    plt.setp(axs_ylabel,size=9,color='black')
    
    #画出散点图,dataSet第二列(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(dataSet[:,1],dataSet[:,2],s=15,color=labelcolors,alpha=0.8)
    axs_title = axs[1][0].set_title('玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs_xlabel = axs[1][0].set_xlabel('玩视频游戏所消耗时间占比',FontProperties=font)
    axs_ylabel = axs[1][0].set_ylabel('每周消费的冰激淋公升数',FontProperties=font,color='black',size=9)
    plt.setp(axs_title,size=9,color='red')
    plt.setp(axs_xlabel,size=9,color='black')
    
    #添加几条线作为图例
    didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntLike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    
    plt.show()
    
file = '机器学习实战/Ch02/datingTestSet2.txt'      
'''      
dataSet,labels = openfile(file)
showdata(dataSet,labels) #绘制数据图
'''

def openfile(filename):
    '''读取文件函数
    Parameters：filename - 文件目录
    Returns：dataSet - 数据集
             labels - 对应的标签
    Author：Li Wei
    '''
    f = open(filename)
    dataSet = []
    labels = []
    for i in f.readlines():
        data = i.strip().split('\t')
        dataSet.append(data[:-1])
        labels.append(data[-1])
    return np.array(dataSet).astype('float'),labels #注意这里要修改数据集的类型，不然会默认为str类型


#观察数据集可以发现，飞行里程数相比其他两个属性值比较大，计算距离的话，会导致其比重比较大，因此将其进行归一
def autoNorm(dataSet):
    '''数据归一化函数
    Parameters：dataSet - 数据集
    Returns：dataSet - 归一化后的数据集
    Author：Li Wei
    '''
    Mindata = dataSet.min(axis=0)
    Maxdata = dataSet.max(axis=0)
    rangedata = Maxdata - Mindata
    dataSet = (dataSet - Mindata) / rangedata  
    return dataSet,Mindata,rangedata #最小值和区间值在后面会用到

#以上，模型建立工作已经完成，下面开始采用数据进行验证
def datingClassTest(filename):
    '''验证函数
    Parameters：filename - 数据集文件目录
    Returns：无
    Author：Li Wei
    '''
    dataSet,labels = openfile(filename)
    dataSet,Mindata,Range = autoNorm(dataSet)
    m = dataSet.shape[0]
    l = int(m*0.1) #采用数据集中10%的数据进行验证
    testdata = dataSet[:l,:] #因为数据集本身就是随机选取的，因此这里可以直接按比例截取
    testlabel = labels[:l]
    
    count = 0
    for i in range(l):
        result = classify(testdata[i,:],dataSet[l:,:],labels[l:],4)
        print('预测结果是{}，实际结果是{}'.format(result,testlabel[i]))
        if result != testlabel[i]:
            count += 1
    print('错误率为{:%}'.format(count/l))
    
'''验证模型
datingClassTest(file)
'''

#下面对程序进行完善，当输入对应人员数据能直接返回结果
def classifyPerson(file):
    '''Knn应用函数
    Parameters：file - 数据集文件
    Returns：无
    Author：Li Wei
    '''        
    resultList = ['讨厌','有些喜欢','非常喜欢']
    ffMiles = float(input("请输入每年获得的飞行常客里程数:"))
    precentTats = float(input('请输入玩视频游戏所耗时间百分比:'))
    iceCream = float(input("请输入每周消费的冰激淋公升数:"))
    
    dataSet,labels = openfile(file)
    dataSet,Mindata,Range = autoNorm(dataSet)
    data = np.array([ffMiles,precentTats,iceCream])
    data = (data - Mindata) / Range
    result = classify(data,dataSet,labels,3)
    print('你可能{}这个人'.format(resultList[int(result)-1])) #数据中，1代表讨厌，2是有些喜欢，3是非常喜欢

if __name__ == '__main__':
    classifyPerson(file)    