import matplotlib.pyplot as plt
from numpy import *

def rssError(yMat, yPred):
    """ Ordinary Least Square Error. """
    return ((yMat - yPred) ** 2).sum()

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
    Locally Weighted Linear Regression algorithm (Gaussian Kernel).

    Args:
        testPoint: Test point, training points around test point are weighted higher.
        xArr: Matrix of training data, sample * feature.
        yArr: Value array of trainig data.
        k: Gaussian kernel parameter.

    Returns:
        Regression weights of each feature.
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

def lwlrTest(testArr, xArr, yArr, k=1.0):
    """ Run lwlr on each point of test array. """ 
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
    """ Run lwlr on each point of test array.  """
    # xMat = mat(xArr); yMat = mat(yArr).T
    # yMean = mean(yMat, 0)
    # yMat = yMat - yMean
    # xMeans = mean(xMat, 0)
    # xVar = var(xMat, 0)
    # xMat = (xMat - xMeans) / xVar
    xMat, yMat = regularize(xArr, yArr)
    wMat = zeros((numIt, shape(xMat)[1]))
    for i in range(numIt):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

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

def stageWise(xMat, yMat, eps=0.01):
    """
    Forward stepwise regression algorithm.

    Args:
        xMat: Matrix of training data, sample * feature.
        yMat: Value array of trainig data.
        eps: Algorithm parameter.

    Returns:
        Regression weights of each feature.
    """
    m, n = shape(xMat)
    ws = zeros((n, 1)); wsTest = ws.copy()
    lowestError = inf
    for j in range(n):
        for sign in [-1, 1]:
            wsTest = ws.copy()
            wsTest[j] += eps * sign
            yTest = xMat * wsTest
            rssE = rssError(yMat.A, yTest.A)
            if rssE < lowestError:
                lowestError = rssE
                ws = wsTest
    return ws

def stageWiseTest(xArr, yArr, eps=0.01, numIt=100):
    """  """
    # xMat = mat(xArr); yMat = mat(yArr).T
    # yMean = mean(yMat, 0)
    # yMat = yMat - yMean
    # xMat = regularize(xMat)
    xMat, yMat = regularize(xArr, yArr)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        ws = stageWise(xMat, yMat, eps)
        # print(ws.T)
        returnMat[i, :] = ws.T
    return returnMat

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

def crossValidation(xArr, yArr, numVal=10):
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

def plot(xMat, yMat, yHat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd]
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # yHat = lwlrTest(xCopy, xArr, yArr, 0.003)
    ax.plot(xSort[:, 1], yHat[srtInd][:, 1])
    plt.show()

if __name__ == '__main__':
    xArr, yArr = loadDataSet('data/ex0.txt')
    xMat = mat(xArr); yMat = mat(yArr)
    ws = standRegres(xArr, yArr); yHat = xMat * ws
    print(corrcoef(yHat.T, yMat))   # correlation coefficient
    yHat = lwlrTest(xArr, xArr, yArr, 1.0)
    plot(xMat, yMat, yHat)
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    plot(xMat, yMat, yHat)
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    plot(xMat, yMat, yHat)


    # abX, abY = loadDataSet('data/abalone.txt')
    # yHat01 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 0.1); print(rssError(abY[0: 99], yHat01.T))
    # yHat1 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 1); print(rssError(abY[0: 99], yHat1.T))
    # yHat10 = lwlrTest(abX[0: 99], abX[0: 99], abY[0: 99], 10); print(rssError(abY[0: 99], yHat10.T))
    # yHat01 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 0.1); print(rssError(abY[100: 199], yHat01.T))
    # yHat1 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 1); print(rssError(abY[100: 199], yHat1.T))
    # yHat10 = lwlrTest(abX[100: 199], abX[0: 99], abY[0: 99], 10); print(rssError(abY[100: 199], yHat10.T))
    # ws = standRegres(abX[0: 99], abY[0: 99]); yHat = mat(abX[100: 199]) * ws
    # print(rssError(abY[100: 199], yHat.T.A))

    
    # abX, abY = loadDataSet('data/abalone.txt')
    # xMat = regularize(abX)
    # yMean = mean(abY, 0)
    # yMat = abY - yMean
    # weights = standRegres(xMat, yMat)
    # print(weights)
    # ridgeWeights = stageWise(abX, abY, 0.001, 5000)
    # # ridgeWeights = ridgeTest(abX, abY)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()

    # lgX, lgY = loadDataSet('data/lego.txt')
    # m, n = shape(lgX)
    # print(m, n)
    # lgX1 = mat(ones((m, n + 1)))
    # lgX1[:, 1: n + 1] = mat(lgX)
    # print(lgX[0])
    # print(lgX1[0])
    # ws = standRegres(lgX1, lgY)
    # print(ws)
    # print(lgX1[0] * ws)
    # print(lgX1[-1] * ws)
    # print(lgX1[43] * ws)
    # crossValidation(lgX, lgY, 10)
    # print(ridgeTest(lgX, lgY))
