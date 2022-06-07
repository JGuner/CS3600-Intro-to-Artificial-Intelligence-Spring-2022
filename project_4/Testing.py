from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)


print ("----------------- testCarData --------------------")
numNeurons = 0
while numNeurons <=0:
    print("--------------running with", numNeurons , "neurons per hidden layer------------------")
    i = 0
    acclist = []
    while i < 5:
	        print("running iteration #", i+1)
	        nnet, testAccuracy = buildNeuralNet(penData,maxItr = 200, hiddenLayerList = [numNeurons])
	        acclist.append(testAccuracy)
	        i = i + 1
    numNeurons += 5
    print ("Iteration finished")
    print ("accuracy average:", average(acclist))
    print ("accuracy standard deviation:", stDeviation(acclist))
    print ("max accuracy:", max(acclist))