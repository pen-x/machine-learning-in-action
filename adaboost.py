from numpy import *

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """ 
    Run a decision stump classifier (one-level decision tree) on training data. 

    Args:
        dataMatrix: Matrix of training data, sample * feature.
        dimen: Index of feature the classifier runs on.
        threshVal: Feature decision boundary.
        threshIneq: Which side of decision boundary is labelled as negative, can be 'lt' or 'gt'.
    
    Returns:
        Array of classifier result for each sample, 1 means positive, -1 means negative.
    """
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataMatrix, classLabels, D):
    """ 
    Find the decision stump with minimum error rate on current sample weight. 
    
    Args:
        dataMatrix: Matrix of training data, sample * feature.
        classLabels: Lable array of trainig data, 1 means positive, -1 means negative.
        D: Weight array of training data.

    Returns:
        bestStump: Best decision stump parameters.
        minError: Minimun weighted error rate.
        bestClassEst: Classify result of best decision stump.
    """
    dataMat = mat(dataMatrix); labelMat = mat(classLabels).T
    m, n = shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n): # Loop through all features
        rangeMin = dataMat[:, i].min(); rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1): # Loop through all split values
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataMatrix, classLabels, numIter=40):
    """
    AdaBoost algorithm main training process.

    Args:
        dataMatrix: Matrix of training data, sample * feature.
        classLabels: Lable array of trainig data, 1 means positive, -1 means negative.
        numIter: Iteration number.

    Returns:
        Array of the best decision stumps on each iteration (weak classifier array).
    """
    weakClassArr = [] # Store the best decision stump on each iteration
    m = shape(dataMatrix)[0]
    D = mat(ones((m, 1)) / m) # Initial each sample to equal weight
    aggClassEst = mat(zeros((m, 1))) # Combined classify result of all best decision stumps
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataMatrix, classLabels, D)
        # print('D:', D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16))) # Make sure won't divide 0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst: ', classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst # Classify result multiple alpha is regarded as partial result
        # print('aggClassEst: ', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print('total error: ', errorRate, '\n')
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    """
    Use trained weak classifier array to classify testing data.

    Args:
        dataToClass: Matrix of testing data.
        classifierArr: Trained weak classifier array.

    Returns:
        Array of classifier result for each sample, 1 means positive, -1 means negative.
    """
    dataMat = mat(dataToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return sign(aggClassEst)

def loadSimpData():
    dataMat = mat([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    # dataMat, classLabels = loadSimpData()
    # classifierArr = adaBoostTrainDS(dataMat, classLabels, 9)
    # classifierResult = adaClassify([[5, 5], [0, 0]], classifierArr)
    # print(classifierResult)

    dataMat, labelMat = loadDataSet('horseColicTraining2.txt')
    classifierArr = adaBoostTrainDS(dataMat, labelMat, 10)
    testMat, testLabelMat = loadDataSet('horseColicTest2.txt')
    prediction = adaClassify(testMat, classifierArr)
    errArr = mat(ones((67, 1)))
    print('error count:', errArr[prediction != mat(testLabelMat).T].sum())