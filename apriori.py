
def createC1(dataSet):
    """ 
    Collect candidate item sets from dataset. Each candidate item contains only one element.
    
    Args:
        dataSet: Input data, array of item list.

    Returns:
        List of candidate item sets.
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])   # each item is wrapped by list
    C1.sort()
    return list(map(frozenset, C1))     # use frozenset to make it unchangable, we can use it as dictionary key later 

def scanD(D, Ck, minSupport):
    """
    Filter out frequent item sets from candidate item sets.

    Args:
        D: Input data, array of item list.
        Ck: Candidate item sets.
        minSupport: Minimun acceptable support (coverage of candidate item set on all records).

    Returns:
        retList: Candidate item sets with coverage higher than minSupport.
        supportData: Dictionary of frequent item sets and their support values.
    """
    ssCnt = {}  # store each candidate item set and its count on all records.
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems     # candidate item set coverage
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    """
    Generate candidate item sets with set size k.

    Args:
        Lk: Candidate item sets with set size k - 1.
        k: Number of elements in each candidate item set.

    Returns:
        Candidate item set with set size k. 
        E.g. [{0}, {1}, {2}] -> [{0, 1}, {0, 2}, {1, 2}]
             [{0, 1}, {0, 2}, {1, 2}] -> [{0, 1, 2}]
    """
    retList = []
    lenLK = len(Lk)
    for i in range(lenLK):
        for j in range(i + 1, lenLK):   # compare each set pair
            L1 = list(Lk[i])[: k - 2]
            L2 = list(Lk[j])[: k - 2]
            L1.sort()
            L2.sort()
            # list(set) is ordered, so we compare the front k-2 elements, if they're same, then only the last element is different, we can combine these two set to get a new set with size k
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    """
    Apriori Algorithm.
    """
    C1 = createC1(dataSet)  # init candidate item sets with set size 1
    L1, supportData = scanD(dataSet, C1, minSupport)    # extract frequent item sets with set size 1
    L = [L1]    # L[0] = L1, L[1] = L2, index is k - 1
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(dataSet, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):
    """
    Generate association rules from frequent item sets

    Args:
        L: Frequent item sets list.
        supportData: Dictionary of frequent item sets and their support values.
        minConf: Minimun acceptable confidence.

    Returns:
        List of association rules with high confidence.
    """
    bigRuleList = []
    for i in range(1, len(L)):  # L[0] contains only single element set, can't generate rules
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]    # split frequent item set into elment list
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:   # frequent item set size is 2, L and R can only contain 1 element
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    """
    Calculate confidence score. Consfidence(L -> R) = support(L | R) / support(L).

    Args:
        freqSet: Frequent item set. E.g. [{1, 2, 3}]
        H: List of candidate rule right item set. E.g. [{1, 2}, {1, 3}]
        supportData: Dictionary of frequent item sets and their support values.
        br1: Store all rules with high confidence.
        minConf: Minimun acceptable confidence.

    Returns:
        List of rule right item set which can generate a rule with high confidence.
    """
    prunedH = []
    for conseq in H:
        # L = conseq, R = freqSet - conseq, L/R = freqSet
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportedData, br1, minConf=0.7):
    """
    Recursive generate rules from frequent item set. 

    Args:
        freqSet: Frequent item set. E.g. [{1, 2, 3}]
        H: List of candidate rule right item set. E.g. [{1, 2}, {1, 3}]
        supportData: Dictionary of frequent item sets and their support values.
        br1: Store all rules with high confidence.
        minConf: Minimun acceptable confidence.
    """
    m = len(H[0])      # rule right item set size
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m + 1)     # generate rule right item set with size m + 1
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1):     # can increase rule right item set size
            rulesFromConseq(freqSet, Hmp1, supportedData, br1, minConf)

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

if __name__ == '__main__':
    dataSet = loadDataSet()
    # C1 = createC1(dataSet); print(C1)
    # L1, supportData0 = scanD(dataSet, C1, 0.5); print(L1)

    L, supportData = apriori(dataSet, minSupport=0.5); print(L)
    rules = generateRules(L, supportData, minConf=0.5)

    # mushDataSet = [line.split() for line in open('data/mushroom.dat').readlines()]
    # L, supportData = apriori(mushDataSet, minSupport=0.3)
    # for item in L[3]:
    #     if item.intersection('2'):
    #         print(item)
