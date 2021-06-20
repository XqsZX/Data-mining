import collections
from math import log
import operator
import treePlotter
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import confusion_matrix


# 创建测试的数据集，里面的数值中具有连续值
def loadData(filename):
    data = pd.read_csv(filename, delimiter=',')
    # 特征值列表
    labels = list(data.columns.values)[:-1]
    labels2 = list(data.columns.values)[:-1]
    # 去除' ?'
    for i in labels:
        data = data[~data[i].isin([' ?'])]
    dataSet = data.values
    trainSet, testSet = train_test_split(dataSet, train_size=0.8, random_state=0)
    trainData = trainSet.tolist()
    testData = testSet.tolist()
    # 特征对应的所有可能的情况
    labels_full = {}
    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel
    return trainData, labels, labels2, labels_full, testData


# 计算给定数据集的信息熵(香农熵)
def calcShannonEnt(dataSet):
    # 计算出数据集的总数
    numEntries = len(dataSet)
    # 统计标签
    labelCounts = collections.defaultdict(int)
    # 循环整个数据集，得到数据的分类标签
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] += 1
    # 初始化信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定的数值，将数据集分为不大于和大于两部分
def splitDataSetForSeries(dataSet, axis, value):
    # 不大于划分值的集合
    lessDataSet = []
    # 大于划分值的集合
    moreDataSet = []
    # 进行划分，保留该特征值
    for feat in dataSet:
        if feat[axis] <= value:
            reducefeat = feat[:axis]
            reducefeat.extend(feat[axis + 1:])
            lessDataSet.append(reducefeat)
        else:
            reducefeat = feat[:axis]
            reducefeat.extend(feat[axis + 1:])
            moreDataSet.append(reducefeat)
    return lessDataSet, moreDataSet


# 按照给定的特征值，将数据集划分
def splitDataSet(dataSet, axis, value):
    # 创建一个新的列表
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 计算连续值的信息增益
def calcInfoGainForSeries(dataSet, i, baseEntropy):
    # 最大的信息增益
    maxInfoGain = 0.0
    # 最好的划分点
    bestMid = -1
    featList = [example[i] for example in dataSet]
    classList = [example[-1] for example in dataSet]
    dictList = dict(zip(featList, classList))
    # 将其从小到大排序，按照连续值的大小排列
    sortedFeatList = sorted(dictList.items(), key=operator.itemgetter(0))
    # 计算连续值的个数
    numberForFeatList = len(sortedFeatList)
    # 计算划分点
    midFeatList = [round((sortedFeatList[i][0] + sortedFeatList[i + 1][0]) / 2.0, 3) for i in range(numberForFeatList - 1)]
    # 计算出各个划分点信息增益
    for mid in midFeatList:
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        lessDataSet, moreDataSet = splitDataSetForSeries(dataSet, i, mid)
        # 计算两部分的特征值熵和权重的乘积之和
        newEntropy = len(lessDataSet) / len(featList) * calcShannonEnt(lessDataSet) + len(moreDataSet) / len(featList) * calcShannonEnt(moreDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # print('当前划分值为：' + str(mid) + '，此时的信息增益为：' + str(infoGain))
        if infoGain > maxInfoGain:
            bestMid = mid
            maxInfoGain = infoGain
    return maxInfoGain, bestMid


# 计算信息增益
def calcInfoGain(dataSet, featList, i, baseEntropy):
    # 将当前特征唯一化，也就是说当前特征值中共有多少种
    uniqueVals = set(featList)
    # 新的熵，代表当前特征值的熵
    newEntropy = 0.0
    # 遍历现在有的特征的可能性
    for value in uniqueVals:
        # 在全部数据集的当前特征位置上，找到该特征值等于当前值的集合
        subDataSet = splitDataSet(dataSet=dataSet, axis=i, value=value)
        # 计算出权重
        prob = len(subDataSet) / float(len(dataSet))
        # 计算出当前特征值的熵
        newEntropy += prob * calcShannonEnt(subDataSet)
    # 计算出信息增益
    infoGain = baseEntropy - newEntropy
    return infoGain


# 选择最好的数据集划分特征，根据信息增益值来计算，可处理连续值
def chooseBestFeatureToSplit(dataSet):
    # 得到数据的特征值总数
    numFeatures = len(dataSet[0]) - 1
    # 计算出基础信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 基础信息增益为0.0
    bestInfoGain = 0.0
    # 最好的特征值
    bestFeature = -1
    # 标记当前最好的特征值是不是连续值
    flagSeries = 0
    # 如果是连续值的话，用来记录连续值的划分点
    bestSeriesMid = 0.0
    # 对每个特征值进行求信息熵
    for i in range(numFeatures):
        # 得到数据集中所有的当前特征值列表
        featList = [example[i] for example in dataSet]
        if isinstance(featList[0], str):
            infoGain = calcInfoGain(dataSet, featList, i, baseEntropy)
        else:
            infoGain, bestMid = calcInfoGainForSeries(dataSet, i, baseEntropy)
        # 如果当前的信息增益比原来的大
        if infoGain > bestInfoGain:
            # 最好的信息增益
            bestInfoGain = infoGain
            # 新的最好的用来划分的特征值
            bestFeature = i
            flagSeries = 0
            if not isinstance(dataSet[0][bestFeature], str):
                flagSeries = 1
                bestSeriesMid = bestMid
    if flagSeries:
        return bestFeature, bestSeriesMid
    else:
        return bestFeature


# 找到次数最多的类别标签
def majorityCnt(classList):
    # 用来统计标签的票数
    classCount = collections.defaultdict(int)
    # 遍历所有的标签类别
    for vote in classList:
        classCount[vote] += 1
    # 从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的标签
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels, i):
    # 拿到所有数据集的分类标签
    classList = [example[-1] for example in dataSet]
    # 统计第一个标签出现的次数，与总标签个数比较，如果相等则说明当前列表中全部都是一种标签，此时停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 计算第一行有多少个数据，如果只有一个的话说明所有的特征属性都遍历完了，剩下的一个就是类别标签,同时设立深度为3
    if len(dataSet[0]) == 1 or i == 4:
        # 返回剩下标签中出现次数较多的那个
        return majorityCnt(classList)
    # 选择最好的划分特征，得到该特征的下标
    bestFeat = chooseBestFeatureToSplit(dataSet=dataSet)
    # 如果是连续值，记录连续值的划分点
    midSeries = 0.0
    # 如果是元组的话，说明此时是连续值
    if isinstance(bestFeat, tuple):
        # 重新修改分叉点信息
        bestFeatLabel = str(labels[bestFeat[0]]) + '小于' + str(bestFeat[1]) + '?'
        # 得到当前的划分点
        midSeries = bestFeat[1]
        # 得到下标值
        bestFeat = bestFeat[0]
        # 连续值标志
        flagSeries = 1
    else:
        # 得到分叉点信息
        bestFeatLabel = labels[bestFeat]
        # 离散值标志
        flagSeries = 0
    # 使用一个字典来存储树结构，分叉处为划分的特征名称
    myTree = {bestFeatLabel: {}}
    # 得到当前特征标签的所有可能值
    featValues = [example[bestFeat] for example in dataSet]
    # 连续值处理
    if flagSeries:
        # 将本次划分的特征值从列表中删除掉
        del (labels[bestFeat])
        # 将连续值划分为不大于当前划分点和大于当前划分点两部分
        eltDataSet, gtDataSet = splitDataSetForSeries(dataSet, bestFeat, midSeries)
        # 得到剩下的特征标签
        subLabels = labels[:]
        # 递归处理小于划分点的子树
        subTree = createTree(eltDataSet, subLabels, i+1)
        myTree[bestFeatLabel]['小于'] = subTree
        subLabels = labels[:]
        # 递归处理大于当前划分点的子树
        subTree = createTree(gtDataSet, subLabels, i+1)
        myTree[bestFeatLabel]['大于'] = subTree
        return myTree
    # 离散值处理
    else:
        # 将本次划分的特征值从列表中删除掉
        del (labels[bestFeat])
        # 唯一化，去掉重复的特征值
        uniqueVals = set(featValues)
        # 遍历所有的特征值
        for value in uniqueVals:
            # 得到剩下的特征标签
            subLabels = labels[:]
            # 递归调用，将数据集中该特征等于当前特征值的所有数据划分到当前节点下，递归调用时需要先将当前的特征去除掉
            subTree = createTree(splitDataSet(dataSet=dataSet, axis=bestFeat, value=value), subLabels, i+1)
            # 将子树归到分叉处下
            myTree[bestFeatLabel][value] = subTree
        return myTree


def test(tree, label2, info):
    root = list(tree)[0]
    firstdict = tree[str(root)]
    data = dict(zip(label2, info))
    div = re.split("小于", root)
    if len(div) == 2:
        if data[div[0]] < float(div[1][:-1]):
            if isinstance(firstdict["小于"], dict):
                label = test(firstdict["小于"], label2, info)
            else:
                label = firstdict["小于"]
        else:
            if isinstance(firstdict["大于"], dict):
                label = test(firstdict["大于"], label2, info)
            else:
                label = firstdict["大于"]
    else:
        if data[div[0]] in firstdict:
            if isinstance(firstdict[data[div[0]]], dict):
                label = test(firstdict[data[div[0]]], label2, info)
            else:
                label = firstdict[data[div[0]]]
        else:
            label = -1
    return label


# 构造随机森林
def buildForest(dataSet):
    trees = []
    for i in range(5):
        data = []
        for j in range(int(0.8 * len(dataSet))):
            data.append(random.randint(0, len(dataSet) - 1))
        data = [dataSet[k] for k in data]
        rand = random.sample(range(14), 5)
        rand.sort(reverse=True)
        label = labels
        for l in rand:
            data = [m[:l] + m[l + 1:] for m in data]
            label = label[:l] + label[l + 1:]
        tree = createTree(data, label, 1)
        treePlotter.createPlot(tree)
        print(tree)
        trees.append(tree)
    return trees


# 进行预测,并对adult数据集结果进行评价
def estimateAdult(testdata, labels2):
    l = len(testdata)
    pre = []
    real = []
    for i in testdata:
        result = []
        for j in trees:
            result.append(test(j, labels2, i[:-1]))
        if majorityCnt(result) == -1:
            l -= 1
        else:
            pre.append(majorityCnt(result))
            real.append(i[-1])
    CM = confusion_matrix(real, pre)
    print("混淆矩阵为： ", CM)
    Error_rate = (CM[0, 1] + CM[1, 0]) / l
    print("误分率：", Error_rate)
    Accuracy = (CM[0, 0] + CM[1, 1]) / l
    print("准确率：", Accuracy)
    Recall = CM[0, 0] / (CM[0, 0] + CM[1, 0])
    print("召回率：", Recall)
    FPR = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    print("FPR：", FPR)
    TPR = CM[0, 0] / (CM[0, 0] + CM[0, 1])
    print("TPR：", TPR)


# 进行预测,并对iris数据集结果进行评价
def estimateIris(testdata, labels2):
    l = len(testdata)
    pre = []
    real = []
    for i in testdata:
        result = []
        for j in trees:
            result.append(test(j, labels2, i[:-1]))
        if majorityCnt(result) == -1:
            l -= 1
        else:
            pre.append(majorityCnt(result))
            real.append(i[-1])
    CM = confusion_matrix(real, pre)
    print("混淆矩阵为： ", CM)
    # 误分率
    Error_rate = (CM[0, 1] + CM[0, 2] + CM[1, 0] + CM[1, 2] + CM[2, 0] + CM[2, 1]) / len(testdata)
    print("误分率：", Error_rate)
    # 准确率
    Accuracy = (CM[0, 0] + CM[1, 1] + CM[2, 2]) / len(testdata)
    print("准确率：", Accuracy)
    # 对于setosa类
    setosa_recall = CM[0, 0] / (CM[0, 0] + CM[1, 0] + CM[2, 0])
    setosa_fpr = (CM[1, 0] + CM[2, 0]) / (CM[1, 0] + CM[2, 0] + CM[1, 1] + CM[2, 1] + CM[1, 2] + CM[2, 2])
    setosa_tpr = CM[0, 0] / (CM[0, 0] + CM[0, 1] + CM[0, 2])
    print("setosa's racall, fpr and tpr are: ", setosa_recall, setosa_fpr, setosa_tpr)
    # 对于versicolor类
    versicolor_recall = CM[1, 1] / (CM[0, 1] + CM[1, 1] + CM[2, 1])
    versicolor_fpr = (CM[0, 1] + CM[2, 1]) / (CM[0, 0] + CM[2, 0] + CM[0, 1] + CM[2, 1] + CM[0, 2] + CM[2, 2])
    versicolor_tpr = CM[1, 1] / (CM[1, 0] + CM[1, 1] + CM[1, 2])
    print("versicolor's racall, fpr and tpr are: ", versicolor_recall, versicolor_fpr, versicolor_tpr)
    # 对于versica类
    versica_recall = CM[2, 2] / (CM[0, 2] + CM[1, 2] + CM[2, 2])
    versica_fpr = (CM[0, 2] + CM[1, 2]) / (CM[0, 0] + CM[1, 0] + CM[0, 1] + CM[1, 1] + CM[0, 2] + CM[1, 2])
    versica_tpr = CM[2, 2] / (CM[2, 0] + CM[2, 1] + CM[2, 2])
    print("versica's racall, fpr and tpr are: ", versica_recall, versica_fpr, versica_tpr)


filename = r"D:\study\综合课程设计\data\adult.csv"
dataSet, labels, labels2, labels_full, testdata = loadData(filename)

trees = buildForest(dataSet)
estimateAdult(testdata, labels2)
