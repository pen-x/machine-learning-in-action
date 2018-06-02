from numpy import *

def regLeaf(dataSet):
    """ Create regression tree leaf, use mean value of data in this leaf as representation. """
    return mean(dataSet[:, -1])

def regErr(dataSet):
    """ Calculate square error on regression tree leaf. """
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def modelLeaf(dataSet):
    """ Create model tree leaf, leaf value is linear regression weights. """
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    """ Calculate square error on model tree leaf. """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

def binSplitDataSet(dataSet, feature, value):
    """
    Split training data according to split feature and split value.

    Args:
        dataSet: Training data.
        feature: Index of split feature.
        value: Split value.

    Returns:
        mat0: Sub-set of training data that has feature value less than split value.
        mat1: Sub-set of training data that has feature value more than split value.
    """
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]  # nonzero returns a tuple of arrays, one for each dimension
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    Create CART (Classification and Regression Tree) for training data.

    Args:
        dataSet: Training data.
        leafType: Method to create tree leaf, regLeaf for regression tree, modelLeaf for model tree.
        errType: Method to calculate leaf error, regErr for regression tree, modelErr for model tree.
        ops = (tolS, tolN): tolS is minimum acceptable delta error, tolN is minimun data count for a leaf node. 

    Returns:
        Tree in dictionary format. 'spInd' and 'spVal' indicate split feature index and value, 'left' and 'right' indicate left and right sub-tree.
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    Find the best split feature index and value for current data set.

    Args:
        dataSet: Training data.
        leafType: Method to create tree leaf, regLeaf for regression tree, modelLeaf for model tree.
        errType: Method to calculate leaf error, regErr for regression tree, modelErr for model tree.
        ops = (tolS, tolN): tolS is minimum acceptable error drop, tolN is minimun data count for a leaf node.

    Returns:
        The best split index and value.
    """
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     # if only one sample left, create one leaf with it, so split feature is None
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)    # S stores error of whole set as tree leaf
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n - 1):
        # each value of this feature would be tested as split value
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:    # log the minimum split error
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:      # if the best split doesn't bring enough error drop, ignore it
        return None, leafType(dataSet)
    # check left and right sub-tree contains enough data
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    """
    Post-prune regression tree on test data.
    """
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if after recursive prune, left and right sub-tree are all leaves, then consider combining them into one leaf
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    """ Standard linear regression algorithm. """
    m, n = shape(dataSet)
    X = mat(ones((m, n))); Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \ntry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def regTreeEval(model, inData):
    """ Evaluate regression tree leaf value. """
    return float(model)

def modelTreeEval(model, inData):
    """ Evaluate model tree leaf value. """
    n = shape(inData)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1: n + 1] = inData
    return float(X * model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    """
    Forecast value for new data sample.

    Args:
        tree: Generated regression tree.
        inData: Feature vector of one sample.
        modelEval: Method to evaluate value on a tree leaf.

    Returns:
        Forecasted value for input data.
    """
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    """ Forcast values on all test data. """
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

if __name__ == '__main__':
    # myMat = mat(loadDataSet('data/ex00.txt')); print(createTree(myMat))
    # myMat = mat(loadDataSet('data/ex0.txt')); print(createTree(myMat, ops=(0, 1)))
    # myMat = mat(loadDataSet('data/ex2.txt')); myMatTest = mat(loadDataSet('data/ex2test.txt'))
    # myTree = createTree(myMat, ops=(0, 1))
    # print(prune(myTree, myMatTest))

    # myMat = mat(loadDataSet('data/exp2.txt'))
    # myTree = createTree(myMat, modelLeaf, modelErr, (1, 10))
    # print(myTree)

    trainMat = mat(loadDataSet('data/bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('data/bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20)) 
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    ws, X, Y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])