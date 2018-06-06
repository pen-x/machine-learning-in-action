class TreeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    
    def inc(self, numOccur):
        self.count += numOccur
    
    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet, minSup=1):
    """
    Create a frequent-pattern tree.

    Args:
        dataSet: Record set.
        minSup: Minimun acceptable support (coverage of candidate item set on all records).

    Returns:
        retTree: Frequent-pattern tree.
        headerTable: Table of frequent item information (count/link).
    """
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):      # use list to copy keys(), otherwise can't delete while iterating
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]     # headerTable[k] links to the first node of k, nodes of same value are linked one by one
    retTree = TreeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}     # store only frequent items from tranSet and sort them by frequency
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]     # headerTable[item][0] is item frequency
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]     # sort (item, count) pairs
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    """
    Update frequent-pattern tree with an item list.

    Args:
        items: List of frequent items.
        inTree: Current frequent-pattern tree.
        headerTable: Table of frequent item information (count/link).
        count: Count of appearance for input frequent item list.
    """
    curItem = items[0]
    if curItem in inTree.children:
        inTree.children[curItem].inc(count)
    else:
        inTree.children[curItem] = TreeNode(curItem, count, inTree)
        # update headerTable link
        if headerTable[curItem][1] == None:
            headerTable[curItem][1] = inTree.children[curItem]
        else:
            updateHeader(headerTable[curItem][1], inTree.children[curItem])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[curItem], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    """ Update header table link. """
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath):
    """ Get path from node to root in frequent-pattern tree. """
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    """ Find all prefix paths ending with basePat. """
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count    # prefix path count is equal to current node count
        treeNode = treeNode.nodeLink    # nodeLink links to the next same value node
    return condPats

def mineTree(inTree, headerTable, minSup, prefix, freqItemList):
    """
    Mine association rules from frequent-pattern tree.

    Args:
        inTree: Current frequent-pattern tree.
        headerTable: Table of frequent item information (count/link).
        minSup: Minimun acceptable support (coverage of candidate item set on all records).
        prefix: Items already used.
        freqItemList: List to store association rules.
    """
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[0])]

    for basePat in bigL:
        newFeqSet = prefix.copy()
        newFeqSet.add(basePat)
        freqItemList.append(newFeqSet)      # prefix + basePat is considered as one association rule
        condPatBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPatBases, minSup)   # create conditional fp-tree 

        if myHead != None:      # mine rules on conditional fp-tree
            print('conditional tree for:', newFeqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFeqSet, freqItemList)
    
def loadSimpData():
    simpData = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simpData

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

if __name__ == '__main__':
    # rootNode = TreeNode('pyramid', 9, None)
    # rootNode.children['eye'] = TreeNode('eye', 13, None)
    # rootNode.children['phoenix'] = TreeNode('phoenix', 3, None)
    # rootNode.disp()

    simpData = loadSimpData()
    initSet = createInitSet(simpData)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    print(findPrefixPath('x', myHeaderTab['x'][1]))
    print(findPrefixPath('z', myHeaderTab['z'][1]))
    print(findPrefixPath('r', myHeaderTab['r'][1]))

    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)

