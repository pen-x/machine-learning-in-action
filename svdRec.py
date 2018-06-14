from numpy import *

def euclidSim(inA, inB):
    """ Euclidean distance similarity of two vectors, mapping to 0 ~ 1. """
    return 1.0 / (1.0 + linalg.norm(inA - inB))

def pearsSim(inA, inB):
    """ Pearson correlation similarity of two vectors, mapping to 0 ~ 1. """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    """ Cosine similarity of two vectors, mapping to 0 ~ 1. """
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeans, item):
    """
    Standard way for estimating user rating score. (Use this user's rating score for other items and similarities between items)

    Args:
        dataMat: Matrix of users' rating score of different product items.
        user, item: Index of user and product item to estimate score.
        simMeans: Method to calculate similarity.
    
    Returns:
        Estimated rating score for certain user of certain product item.
    """
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # list of users (index) that has scored both product item and j
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # calculate similarity between product item and j
            similarity = simMeans(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
 
def svdEst(dataMat, user, simMeans, item):
    """
    Estimating user rating score using SVD.

    Args:
        dataMat: Matrix of users' rating score of different product items.
        user, item: Index of user and product item to estimate score.
        simMeans: Method to calculate similarity.
    
    Returns:
        Estimated rating score for certain user of certain product item.
    """
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U, Sigma, VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[: 4])     # for our data, top 4 elements contain more than 90% data energy
    xFormedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeans(xFormedItems[item, :].T, xFormedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal
   
def recommend(dataMat, user, N=3, simMeans=cosSim, estMethod=standEst):
    """ Recommend top N unrated items for user. """
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeans, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda j: j[1], reverse=True)[: N]

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print('1', end=' ')
            else:
                print('0', end=' ')
        print('')

def imgCompress(numSV=3, thresh=0.8):
    my1 = []
    for line in open('data/0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        my1.append(newRow)
    myMat = mat(my1)
    print('**** original matrix ****')
    printMat(myMat, thresh)
    U, Sigma, VT = linalg.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print('**** reconstructed matrix using %d singular values ****' % numSV)
    printMat(reconMat, thresh)

def loadExData():
    return [
        [0, 0, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 1, 0, 0]
    ]

def loadExData2():
    return [
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
        [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
        [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
        [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
    ]
    
if __name__ == '__main__':
    # Data = loadExData()
    # U, Sigma, VT = linalg.svd(Data); print(Sigma)
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # Data2 = U[:, :3] * Sig3 * VT[:3, :]; print(Data2)

    # myMat = mat(loadExData())
    # print(euclidSim(myMat[:, 0], myMat[:, 4])); print(euclidSim(myMat[:, 0], myMat[:, 0]))
    # print(cosSim(myMat[:, 0], myMat[:, 4])); print(cosSim(myMat[:, 0], myMat[:, 0]))
    # print(pearsSim(myMat[:, 0], myMat[:, 4])); print(pearsSim(myMat[:, 0], myMat[:, 0]))

    # myMat = mat(loadExData())
    # myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4; myMat[3, 3] = 2
    # print(recommend(myMat, 2))
    # print(recommend(myMat, 2, simMeans=euclidSim))
    # print(recommend(myMat, 2, simMeans=pearsSim))

    # Data = loadExData2()
    # U, Sigma, VT = linalg.svd(mat(Data)); print(Sigma)
    # Sig2 = Sigma ** 2
    # print(sum(Sig2) * 0.9); print(sum(Sig2[: 3]))

    # myMat = mat(loadExData2())
    # print(recommend(myMat, 1, estMethod=svdEst))
    # print(recommend(myMat, 1, estMethod=svdEst, simMeans=pearsSim))

    imgCompress(2)
