import matplotlib.pyplot as plt
from numpy import *

def plotROC(predStrengths, classLabels):
    """
    Plot roc curve and calculate auc.

    Args:
        predStrengths: Classifier predicted value array of training data. (e.g. possibility for naive bayes, value before sign for adaboost).
        classLabels: Lable array of trainig data, 1 means positive, -1 means negative.

    Returns:
        roc curve: X-axis is FP / (FP + TN), y-axis is TP / (TP + FN).
        auc: Area under roc curve.
    """
    cur = (1.0, 1.0)    # point position to draw, start at top right
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)   # yStep = 1 / (TP + FN)
    xStep = 1 / float(len(classLabels) - numPosClas)    # xStep = 1 / (FP + TN)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # Assume original threshold is 0, means every sample is marked as positive, so (x, y) is (1.0, 1.0),
    # slowly increase threshold so each time only one sample is moved from positive to negative,
    # since FP + TN and TP + FN is always the same, it's easy to recalculate (x, y).
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:   # if label is positive, TP = TP - 1
            delX = 0; delY = yStep
        else:   # if label is negative, FP = FP - 1
            delX = xStep; delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], color='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positivie Rate')
    plt.title('ROC curve')
    ax.axis([0, 1, 0, 1])
    plt.show()
    auc = ySum * xStep      # AUC equals the sum area of many small rectangle, each rectange area = xStep * y or yStep * x.
    print('The area under the curve is:', auc)
