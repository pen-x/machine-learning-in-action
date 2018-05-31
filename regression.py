import matplotlib.pyplot as plt
from numpy import *

def standRegres(xArr, yArr):
    """
    Standard linear regression algorithm.

    Args:
        xArr: Matrix of training data, sample * feature.
        yArr: Value array of trainig data.
    
    Returns:
        Regression weights of each feature.
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    Locally Weighted Linear Regression algorithm on single test point (Gaussian Kernel).

    Args:
        testPoint: Test point, training points around test point are weighted higher.
        xArr: Matrix of training data, sample * feature.
        yArr: Value array of trainig data.
        k: Gaussian kernel parameter.

    Returns:
        Regression weights of each feature based on test point.
    """
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))     # size: m * m
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))  # use Gaussian kernel to calculate weight for point j
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return ws

def lwlrRegres(testArr, xArr, yArr, k=1.0):
    """ 
    Locally Weighted Linear Regression algorithm.

    Args:
        testArr: Matrix of test data, sample * feature.
        xArr: Matrix of training data, sample * feature.
        yArr: Value array of trainig data.
        k: Gaussian kernel parameter.

    Returns:
        Value array of test data.
    """ 
    m = shape(testArr)[0]
    yHat = zeros((m, 1))
    for i in range(m):  # for each test point, we calcluate a unique regression weights.
        yHat[i] = testArr[i] * lwlr(testArr[i], xArr, yArr, k)
    return yHat

def ridgeRegres(xMat, yMat, lam=0.2):
    """
    Ridge regression algorithm.

    Args:
        xMat: Matrix of training data, sample * feature.
        yMat: Value array of trainig data.
        lam: Algorithm parameter.

    Returns:
        Regression weights of each feature. 
    """
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr, numIt=30):
    """ 
    Run lwlr on each point of test array. 

    Returns:
        History of weights change. 
    """
    xMat, yMat = regularize(xArr, yArr)     # training data need to be regularized first
    wMat = zeros((numIt, shape(xMat)[1]))
    for i in range(numIt):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def stageWiseRegres(xArr, yArr, eps=0.01, numIt=100):
    """
    Forward stepwise regression algorithm.

    Args:
        xMat: Matrix of training data, sample * feature.
        yMat: Value array of trainig data.
        eps: Algorithm parameter.

    Returns:
        wsMax: Regression weights of each feature.
        returnMat: History of weights change.
    """
    xMat, yMat = regularize(xArr, yArr)     # training data need to be regularized first
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        # print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return wsMax, returnMat

def rssError(yMat, yPred):
    """ Ordinary Least Square Error. """
    return ((yMat - yPred) ** 2).sum()

def regularize(xArr, yArr):
    """
    Data normalization process.

    Args:
        xArr: Matrix of data, sample * feature.
        yArr: Value array of data.

    Returns:
        Normalized result.
    """
    xMat = mat(xArr)
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar 
    
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    
    return xMat, yMat

def crossValidation(xArr, yArr, numVal=10):
    """ Run 10-folder cross-validation with ridgeTest. """
    m = len(yArr)
    indexList = list(range(m))
    errorMat = zeros((numVal, 30))
    for i in range(numVal):
        trainX = []; trainY = []
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX); matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr); yMat = mat(yArr).T
    meanX = mean(xMat, 0); varX = var(xMat, 0)
    unReg = bestWeights / varX
    print('the best model from Ridge Regression is:\n', unReg)
    print('with constant term:', -1 * sum(multiply(meanX, unReg)) + mean(yMat))

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def plotBestFit(xMat, yMat, yHat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[:, 1][srtInd]
    ax.plot(xSort[:, 0], yHat[srtInd][:, 0])
    plt.show()

def plotWeightChange(wsHis):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wsHis)
    plt.show()

if __name__ == '__main__':
    # xArr, yArr = loadDataSet('data/ex0.txt')
    # xMat = mat(xArr); yMat = mat(yArr)
    # ws = standRegres(xArr, yArr); yHat = xMat * ws
    # print(corrcoef(yHat.T, yMat))   # correlation coefficient
    # plotBestFit(xMat, yMat, yHat)
    # yHat = lwlrRegres(xArr, xArr, yArr, 1.0)
    # plotBestFit(xMat, yMat, yHat)
    # yHat = lwlrRegres(xArr, xArr, yArr, 0.01)
    # plotBestFit(xMat, yMat, yHat)
    # yHat = lwlrRegres(xArr, xArr, yArr, 0.003)
    # plotBestFit(xMat, yMat, yHat)


    # abX, abY = loadDataSet('data/abalone.txt')
    # yHat01 = lwlrRegres(abX[0: 99], abX[0: 99], abY[0: 99], 0.1); print(rssError(abY[0: 99], yHat01.T))
    # yHat1 = lwlrRegres(abX[0: 99], abX[0: 99], abY[0: 99], 1); print(rssError(abY[0: 99], yHat1.T))
    # yHat10 = lwlrRegres(abX[0: 99], abX[0: 99], abY[0: 99], 10); print(rssError(abY[0: 99], yHat10.T))
    # yHat01 = lwlrRegres(abX[100: 199], abX[0: 99], abY[0: 99], 0.1); print(rssError(abY[100: 199], yHat01.T))
    # yHat1 = lwlrRegres(abX[100: 199], abX[0: 99], abY[0: 99], 1); print(rssError(abY[100: 199], yHat1.T))
    # yHat10 = lwlrRegres(abX[100: 199], abX[0: 99], abY[0: 99], 10); print(rssError(abY[100: 199], yHat10.T))
    # yHat = mat(abX[100: 199]) * standRegres(abX[0: 99], abY[0: 99]); print(rssError(abY[100: 199], yHat.T.A))

    
    # abX, abY = loadDataSet('data/abalone.txt')
    # xMat, yMat = regularize(abX, abY)
    # ws = standRegres(abX, abY); print(ws)
    # ws, wsHis = stageWiseRegres(abX, abY, 0.001, 5000); plotWeightChange(wsHis)
    # wsHis = ridgeTest(abX, abY); plotWeightChange(wsHis)
    

    lgX, lgY = loadDataSet('data/lego.txt')
    m, n = shape(lgX)
    lgX1 = mat(ones((m, n + 1)))
    lgX1[:, 1: n + 1] = mat(lgX)
    ws = standRegres(lgX1, lgY)
    print('the best model from Ridge Regression is:\n', ws.T)
    crossValidation(lgX, lgY, 10)
