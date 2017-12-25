from math import log
import pandas as pd 
from numpy import *
import time
import random
import sys
from copy import deepcopy

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

class TreeNode(object):
    def __init__(self, value):
        self.left = None
        self.right = None
        self.nodeNumber = 0
        self.nodeAttribute = value
        self.positiveLabels = 0
        self.negativeLabels = 0

def getEntropy(count1, count2):
    if(count1 == 0 or count2 ==0):
        return 0
    else:
        a = true_divide(count1,(count1+count2)) 
        b = true_divide(count2,(count1+count2))   
        return (-a * math.log(a,2) - b * math.log(b,2))

def getBestAttribute(attributes,df):  
    selectedAttribute = ""
    selectedInformationGain = -1
    parentPositiveLabels = len(df.loc[df['Class'] == 1])
    parentNegativeLabels = len(df.loc[df['Class'] == 0])
    parentTotalLabels = parentPositiveLabels + parentNegativeLabels
    parentEntropy = getEntropy(parentPositiveLabels,parentNegativeLabels)

    for i in range(0,len(attributes)):       
        attr = attributes[i]
        filterZero = df.loc[df[attr] == 0]
        filterOne = df.loc[df[attr] == 1]
        
        leftChildpositiveLabels =  len(filterZero.loc[filterZero['Class'] == 1])
        leftChildnegativeLabels =  len(filterZero.loc[filterZero['Class'] == 0])
        leftChildTotalLabels = leftChildpositiveLabels + leftChildnegativeLabels
        
        leftChildEntropy = getEntropy(leftChildpositiveLabels,leftChildnegativeLabels)
        
        rightChildpositiveLabels =  len(filterOne.loc[filterOne['Class'] == 1])
        rightChildnegativeLabels =  len(filterOne.loc[filterOne['Class'] == 0])
        rightChildTotalLabels = rightChildpositiveLabels + rightChildnegativeLabels

        rightChildEntropy = getEntropy(rightChildpositiveLabels,rightChildnegativeLabels)

        weightedEntropy = true_divide(leftChildTotalLabels,parentTotalLabels)*leftChildEntropy +  true_divide(rightChildTotalLabels,parentTotalLabels)*rightChildEntropy

        informationGain = parentEntropy - weightedEntropy

        if(informationGain > selectedInformationGain):
            selectedInformationGain = informationGain
            selectedAttribute = attr
          
    return selectedAttribute


def buildTree(attributes, filteredData):
    global nodeNum
    attr = getBestAttribute(attributes,filteredData)
    
    # New Tree Node
    root = TreeNode(attr)
    nodeNum = nodeNum + 1
    root.nodeNumber = nodeNum
    root.positiveLabels = filteredData['Class'].sum()
    root.negativeLabels = len(filteredData) - root.positiveLabels

    filterDataLeft = filteredData.loc[filteredData[attr] == 0].drop(attr,1)
    filterDataRight = filteredData.loc[filteredData[attr] == 1].drop(attr,1)
    attributes.remove(attr)
    leftPosLabels = filterDataLeft['Class'].sum()
    leftNegLabels = len(filterDataLeft) - leftPosLabels
    
    if(leftPosLabels == 0):
        root.left = TreeNode(0)
        nodeNum = nodeNum + 1
        root.left.nodeNumber = nodeNum
        root.left.negativeLabels = leftNegLabels
    elif(leftNegLabels == 0):
        root.left = TreeNode(1)
        nodeNum = nodeNum + 1
        root.left.nodeNumber = nodeNum
        root.left.positiveLabels = leftPosLabels
    elif(len(attributes) == 0):
        root.left = TreeNode(1) if leftPosLabels > leftNegLabels else TreeNode(0)
        nodeNum = nodeNum + 1
        root.left.nodeNumber = nodeNum
        root.left.positiveLabels = leftPosLabels
        root.left.negativeLabels = leftNegLabels
    else:
        root.left = buildTree(list(attributes),filterDataLeft)

    rightPosLabels = filterDataRight['Class'].sum()
    rightNegLabels = len(filterDataRight) - rightPosLabels
       
    if(rightPosLabels == 0):
        root.right = TreeNode(0)
        nodeNum = nodeNum + 1
        root.right.nodeNumber = nodeNum
        root.right.negativeLabels = rightNegLabels
    elif(rightNegLabels == 0):
        root.right = TreeNode(1)
        nodeNum = nodeNum + 1
        root.right.nodeNumber = nodeNum
        root.right.positiveLabels = rightPosLabels
    elif(len(attributes) == 0):
        root.right = TreeNode(1) if rightPosLabels > rightNegLabels else TreeNode(0)
        nodeNum = nodeNum + 1
        root.right.nodeNumber = nodeNum
        root.right.positiveLabels = rightPosLabels
        root.right.negativeLabels = rightNegLabels
    else:
        root.right = buildTree(list(attributes),filterDataRight)

    return root

def getLeafNodesCount(root):
    if root is not None:
        if(root.nodeAttribute == 0 or root.nodeAttribute == 1):
            return 1
        else:
            return getLeafNodesCount(root.left) + getLeafNodesCount(root.right)
        

def getTotalNodesCount(root):
        if(root is None):
            return 0
        else:
            return 1+ getTotalNodesCount(root.left) + getTotalNodesCount(root.right)

def getClass(root, row):
    if(root.nodeAttribute == 0 or root.nodeAttribute == 1):
        return root.nodeAttribute
    attr = getattr(row,root.nodeAttribute)
    if(attr == 0):
        val = getClass(root.left,row)
    elif(attr == 1):
        val = getClass(root.right,row)
    return val

def getAccuracy(root,dataset,type):
    dataFrame = pd.read_csv(dataset)
    buildClassSet =[]
    inputClassSet = []
    for row in dataFrame.itertuples(index=False, name="Input"):
        buildClassSet.append(getClass(root, row))
        inputClassSet.append(row.Class)
    match = 0
    for i in range(0,len(buildClassSet)):
        if(buildClassSet[i] == inputClassSet[i]):
            match += 1 
    if type != 'internal':
        print "Number of " + type +" instances = ",len(dataFrame)
        print "Number of " + type +" attributes = ",len(dataFrame.columns.values.tolist()) - 1
        if type == 'training':
            print "Total number of nodes in the tree = ",getTotalNodesCount(root)
            print "Number of leaf nodes in the tree = ",getLeafNodesCount(root)
        print "Accuracy of model on the " + type +" dataset = ", round(true_divide(match,len(dataFrame)) * 100,2),"%"
        print

    return round(true_divide(match,len(dataFrame)) * 100,2)

def prune(root, nodeNumber):
    if root is not None:
        #print root.nodeNumber
        if(root.left is None and root.right is None):
            return False
        if(root.nodeNumber == nodeNumber and root.nodeAttribute != 0 and root.nodeAttribute != 1):
            #print "Captured"
            root.left = None
            root.right = None
            root.nodeAttribute = 1 if root.positiveLabels >= root.negativeLabels else 0
            return True
        return prune(root.left, nodeNumber) or prune(root.right, nodeNumber)
    return False    

def pruning(root, pruningFactor, validationPath):   
    totalNode = getTotalNodesCount(root)
    nodesToPrune = pruningFactor*totalNode
    accuracyBeforePruning = getAccuracy(root,validationPath,'internal')
    improvedAccuracy = False
    while(not(improvedAccuracy)):
        node = deepcopy(root)
        count = 0
        while(count < nodesToPrune):
            nodeNumber = random.randrange(totalNode/2,totalNode)
            #print "Node is", nodeNumber
            success = prune(node, nodeNumber)
            if(success):
                count += 1
            totalNode = getTotalNodesCount(node)
            #print "Total Nodes ",totalNode
        prunedAccuracy = getAccuracy(node,validationPath,'internal')
        if prunedAccuracy > accuracyBeforePruning :
            improvedAccuracy = True
    return node


def printTree(root,sep):
    if(root != None):
        if(root.left != None and (root.left.nodeAttribute != 0 and root.left.nodeAttribute != 1)):
            print ' '.join(sep),root.nodeAttribute,"=","0",":"
            sep.append('|')
            printTree(root.left,list(sep))
            sep.remove('|')
        elif(root.left != None and (root.left.nodeAttribute == 0 or root.left.nodeAttribute == 1)):
            print ' '.join(sep),root.nodeAttribute,"=","0",":",root.left.nodeAttribute
            
        if(root.right != None and (root.right.nodeAttribute != 0 and root.right.nodeAttribute != 1)):
            print ' '.join(sep),root.nodeAttribute,"=","1",":"
            sep.append('|')
            printTree(root.right,list(sep))
        elif(root.right != None and (root.right.nodeAttribute == 0 or root.right.nodeAttribute == 1)):
            print ' '.join(sep),root.nodeAttribute,"=","1",":",root.right.nodeAttribute
            



def assignLevelNumbers(root):
    count = 0
    q = Queue()
    if(root is not None):
        q.enqueue(root)
    while not(q.isEmpty):
        node = q.dequeue()
        count += 1
        node.nodeNumber = count 
        if(node.left is not None):
            q.enqueue(node.left)
        if(node.right is not None):
            q.enqueue(node.right)
        


def averageDepth(root, count):
    global leavesDepthCount
    if root is not None:
        if(root.left is None and root.right is None):
            leavesDepthCount += count 
            return 
        averageDepth(root.left,count + 1)
        averageDepth(root.right,count + 1)
    
def main(args):
    trainingPath = args[1]
    validationPath = args[2]
    testingPath = args[3]
    pruningFactor = args[4]

    df = pd.read_csv(trainingPath)
    attributes = df.columns.values.tolist()
    attributes.remove('Class')
    global nodeNum
    nodeNum = 0
    global leavesDepthCount
    leavesDepthCount = 0
    
    start_time = time.time()
    root  =  buildTree(attributes,df)
    print
    print "Time taken to build the tree is ",("--- %s seconds ---" % (time.time() - start_time))
    print
    print "No pre-processing of data is required"
    print "Building a decision tree....Please wait......"


    print 
    print "Decision Tree:"
    print "-------------------------------------------------------------------"
    print
    printTree(root,[])
    print

    print "Pre-Pruned Accuracy"
    print "-------------------------------------------------------------------"
    getAccuracy(root,trainingPath,'training')
    getAccuracy(root,validationPath,'validation')
    getAccuracy(root,testingPath,'testing')
    assignLevelNumbers(root)
    print "-------------------------------------------------------------------"

    #averageDepth(root,0)
    #print
    #print "Average depth of the tree is ", true_divide(leavesDepthCount,getLeafNodesCount(root))
    #print 

    #Pruning the tree 
    print "Pruning the tree... Please wait...."

    root = pruning(root,0.03,validationPath)
    print 
    print "Post-Pruned Decision Tree:"
    print "-------------------------------------------------------------------"
    print
    printTree(root,[])
    print

    print "Post-Pruned Accuracy"
    print "-------------------------------------------------------------------"
    getAccuracy(root,trainingPath,'training')
    getAccuracy(root,validationPath,'validation')
    getAccuracy(root,testingPath,'testing')
    print "-------------------------------------------------------------------"
    


main(sys.argv)