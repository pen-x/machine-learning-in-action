import matplotlib
import matplotlib.pyplot as plt
from numpy import *

def distEclud(vecA, vecB):
    """ Euclidean distance of two vectors. """
    return sqrt(sum(power(vecA - vecB, 2)))

def distSLC(vecA, vecB):
    """ Calculate distance of two points on earth surface. """
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0

def randCent(dataSet, k):
    """ Create k random centroids of training data. """
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    """
    K-Means clustering algorithm.

    Args:
        dataSet: Training data.
        k: Number of clusters.
        distMeans: Method to calculate distance between two data points.
        createCent: Method to initialize cluster centroids.

    Returns:
        centroids: Each cluster's centroid.
        clusterAssment: Cluster index and square distance to cluster centroid for each data.
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2))) 
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        for cent in range(k):   # re-calculate centroids for each cluster
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeans=distEclud):
    """
    Bisecting K-Means clustering algorithm.

    Args:
        dataSet: Training data.
        k: Number of clusters.
        distMeans: Method to calculate distance between two data points.

    Returns:
        centList: Each cluster's centroid.
        clusterAssment: Cluster index and square distance to cluster centroid for each data.
    """
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2))) 

    # there's only one cluster of whole data set in the beginning
    centroid0 = mean(dataSet, axis=0).tolist()[0]   
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :]) ** 2

    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsCurrCluster, 2, distMeans)
            sseSplit = sum(splitClustAss[:, 1])     # sum square error of cluster i splitting into two clusters
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])   # sum square error of clusters other than i
            print('sseSplit, and notSplit:', sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # update best split result index, 0 -> bestCentToSplit, 1 -> next unused number
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is:', bestCentToSplit)
        print('the len of bestClustAss is:', len(bestClustAss))

        # use best split result to update centroids and clusterAssment
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment 

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat
    
def clusterClubs(numClust=5):
    dataList = []
    for line in open('data/places.txt').readlines():
        lineArr = line.split('\t')
        dataList.append([float(lineArr[4]), float(lineArr[3])])

    dataMat = mat(dataList)
    myCentroids, clustAssing = biKmeans(dataMat, numClust, distMeans=distSLC)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarker = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('data/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = dataMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarker[i % len(scatterMarker)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()    

if __name__ == '__main__':
    # dataMat = mat(loadDataSet('data/testSet.10.txt'))
    # myCentroids, clustAssing = kMeans(dataMat, 4)
    # print(myCentroids)

    # dataMat = mat(loadDataSet('data/testSet2.10.txt'))
    # centList, myNewAssments = biKmeans(dataMat, 3)
    # print(centList)

    clusterClubs(5)
