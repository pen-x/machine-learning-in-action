from math import log
import operator

def calcShannonEnt(dataSet):
    """ Calculate shannon entropy of dataSet. """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """ 
    Filter dataSet by feature value.

    Args:
        dataSet: List of data to filter.
        axis: Index of feature to filter.
        value: Value of feature to filter.

    Returns:
        List of data matching feature value.
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # jump over split feature
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """ Find the best feature to split data set. """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)   # entropy of whole set
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """ Find the majoirty label from label list. """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    Create ID3 decision tree.

    Args:
        dataSet, labels: feature and label list of training data.

    Returns:
        Decision tree in dictionary format.
    """
    classList = [example[-1] for example in dataSet]
    # case 1: all labels are same
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # case 2: no features are left
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]   # copy remaining labels to avoid conflict
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)    # recursively create sub-tree
    return myTree

def classify(inputTree, featLabels, testVec):
    """
    Classify testVec to get its label.

    Args:
        inputTree: Created decision tree.
        featLabels: List of available labels, use to match index.
        testVec: Feature vector of test data.

    Returns:
        Matched label.
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def createDataSet():
    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

if __name__ == '__main__':
    # myDat, labels = createDataSet()
    # print(myDat)
    # print(calcShannonEnt(myDat))

    # myDat[0][-1] = 'maybe'
    # print(myDat)
    # print(calcShannonEnt(myDat))

    # print(splitDataSet(myDat, 0, 1))
    # print(splitDataSet(myDat, 0, 0))
    # print(chooseBestFeatureToSplit(myDat))
    # myTree = createTree(myDat, labels[:])
    # print(myTree)


    # print(classify(myTree, labels, [1, 0]))
    # print(classify(myTree, labels, [1, 1]))

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    import treePlotter
    treePlotter.createPlot2(lensesTree)
