import os
import sys
import json
import time
import pyspark
from operator import add
from copy import deepcopy
from pprint import pprint
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task2')
sc.setLogLevel("ERROR")

## Defining the path to the data. 
dataSetPath = sys.argv[1]

## Output file path1.
outfilePath1 = sys.argv[2]

## Output file path2.
outfilePath2 = sys.argv[3]

## Loading the raw data.
rawData = sc.textFile(dataSetPath).map(lambda f: f.split(" "))

## Create the mapping.
graphVertices = set(rawData.map(lambda f: f[0]).collect() + rawData.map(lambda f: f[1]).collect())
graphEdges = rawData.map(lambda f : (f[0], f[1])).collect()
numEdges = len(graphEdges)

## Initialising the adjcacency list for the graph.
adjcacencyList = defaultdict(set)

## Populate the adjcacencyList.
for edge in graphEdges:

	## Extract the ends of the edges.
	firstNode, secondNode = edge[0], edge[1]

	## Add edges according to an undirected graph representation.
	adjcacencyList[firstNode].add(secondNode)
	adjcacencyList[secondNode].add(firstNode)

## pprint(adjcacencyList)

## Function to perform BFS.
def performBFS(rootNode, adjcacencyList):

	## Initialise a visitedSet for avoiding cycles.
	visitedSet = set()
	visitedList = []
	visitedSet.add(rootNode)

	## Initialise a dictionary to hold the parents.
	parentDict = defaultdict(set)

	## Initialise a dictionary to hold the level in the bfsTree.
	currDepth = 0
	levelDict = defaultdict(int)
	levelDict[rootNode] = currDepth

	## Initialise a dictionary to hold the shortest paths 
	## to a node in the bfsTree.
	shortestPaths = defaultdict(int)
	shortestPaths[rootNode] = 1

	## Initialise bfsQ.
	bfsQ = []
	bfsQ.append(rootNode)

	while (len(bfsQ) != 0):

		## Extract the current node.
		currNode = bfsQ.pop(0)
		visitedList.append(currNode)

		## Extract all the neighbours.
		neighbourList = adjcacencyList[currNode]

		## Add neighbours to the queue.
		for neighbour in neighbourList:

			## Check if neighbour has not been visited.
			if (neighbour not in visitedSet):
				levelDict[neighbour] = levelDict[currNode] + 1
				parentDict[neighbour].add(currNode)
				shortestPaths[neighbour] += shortestPaths[currNode]
				visitedSet.add(neighbour)
				bfsQ.append(neighbour)

			## If neighbour has been visited.
			else:

				## Check if the current node is not an adjcacent neighbour.
				if (levelDict[currNode] + 1 == levelDict[neighbour]):
					shortestPaths[neighbour] += shortestPaths[currNode]
					parentDict[neighbour].add(currNode)
	
	return levelDict, parentDict, shortestPaths, visitedList

# Function to edge cost.
def computeEdgeCost(rootNode, adjcacencyList):

	## Perform BFS.
	levelDict, parentDict, shortestPaths, visitedList = performBFS(rootNode, adjcacencyList)

	## Initialise a dictionary to store all the edge costs.
	edgeCost = defaultdict(float)

	## Initialise the node cost for all the elements.
	nodeCost = defaultdict(float)
	for nodeElem in visitedList:
		nodeCost[nodeElem] = 1.0

	## Traversing the nodes.
	for currNode in visitedList[::-1]:

		## Update value for all the parents of the current node.
		parentList = parentDict[currNode]
		for currParent in parentList:
			currContrib = nodeCost[currNode] * float(shortestPaths[currParent] / shortestPaths[currNode])
			edgeCost[tuple(sorted([currParent, currNode]))] += currContrib
			nodeCost[currParent] += currContrib

	return edgeCost

## Function to compute the between-ness.
def computeBetweennes(graphVertices, adjcacencyList):

	## Initialise a dictionary to hold the edge-between-ness.
	edgeBetweennes = defaultdict(int)

	## Choose each node as a vertex once.
	for rootNode in graphVertices:
		edgeCost = computeEdgeCost(rootNode, adjcacencyList)

		## Populate the values for the edge-betweenness.
		for edgeName, edgeCost in edgeCost.items():
			edgeBetweennes[edgeName] += float(edgeCost / 2)

	## Converting the edge betweeness to a sorted list.
	edgeBetweennesList = []
	for key, value in edgeBetweennes.items():
		keyCopy = list(key)
		edgeBetweennesList.append(((keyCopy[0], keyCopy[1]), value))

	## Sort the list.
	edgeBetweennesList.sort(key = lambda f : (-f[1], f[0]))
	return edgeBetweennesList

## Function for creating a set of communities by removing an edge.
def createCommunities(dynamicGraph, vertexPair):

	## Create a copy of the graph.
	currGraph = deepcopy(dynamicGraph)

	## Extract the edge endpoints.
	nodeA = vertexPair[0]
	nodeB = vertexPair[1]

	## Remove the edge from the graph.
	currGraph[nodeA].remove(nodeB)
	currGraph[nodeB].remove(nodeA)

	## Initialise the connected components in the new graph.
	allCC = []

	## Container to hold all the vertices.
	allVertices = set(currGraph.keys())

	## Find the connected components.
	while (len(allVertices) != 0):

		## Extract the current vertex.
		currVertex = allVertices.pop()

		## Initialise a visitedSet for avoiding cycles.
		visitedSet = set()
		visitedSet.add(currVertex)

		## Initialise bfsQ.
		bfsQ = []
		bfsQ.append(currVertex)

		while (len(bfsQ) != 0):

			## Extract the current node.
			currNode = bfsQ.pop(0)

			## Extract all the neighbours.
			neighbourList = currGraph[currNode]

			## Add neighbours to the queue.
			for neighbour in neighbourList:

				## Check if neighbour has not been visited.
				if (neighbour not in visitedSet):

					visitedSet.add(neighbour)
					allVertices.remove(neighbour)
					bfsQ.append(neighbour)

		## A connected component has been found.
		allCC.append(sorted(list(visitedSet)))

	return allCC		

## Function for computing the modularity.
def computeModularity(adjcacencyList, communitySet, numEdges):

	## Initialise the value for modularity.
	totModularity = 0.0

	## Iterate over all communities.
	for currComm in communitySet:

		## Iterate over all node pairs.
		for i in currComm:
			for j in currComm:

				## Check whether an edge between the current nodes exists or not.
				nodeExist = 0.0
				if (i in adjcacencyList[j]):
					nodeExist = 1.0

				## Degrees of the nodes in the original graph.
				degA = len(adjcacencyList[i])
				degB = len(adjcacencyList[j])

				totModularity += nodeExist - ((degA * degB)/ (2 * numEdges))

	## Normalize modularity.
	totModularity /= 2 * numEdges
	return totModularity

## Function for identifying the optimal set of communities.
def optCommunities(graphVertices, adjcacencyList, numEdges, logFile):

	## Compute the betweenness on the original graph.
	edgeBetweennes = computeBetweennes(graphVertices, adjcacencyList)

	## Create a copy of the original graph.
	dynamicGraph = deepcopy(adjcacencyList)

	## Global tracking variables.
	maxModularity = computeModularity(adjcacencyList, [graphVertices], numEdges)
	optCommunitySet = []
	
	## Loop till each community is a single node.
	while (edgeBetweennes):

		## Extract the edge with the maximum betweenness.
		removedEdge = edgeBetweennes[0][0]

		## Compute the set of communities created by removing the current edge.
		communityList = createCommunities(dynamicGraph, removedEdge)

		## Creating a communitySet.
		communitySet = deepcopy(communityList)
		communitySet = [set(i) for i in communityList]

		## Compute the modularity for the community set.
		modularityVal = computeModularity(adjcacencyList, communitySet, numEdges)

		## Check whether this is the optimal community set.
		if (modularityVal > maxModularity):
			maxModularity = modularityVal
			optCommunitySet = deepcopy(communityList)

		## Remove the edge from the graph.
		dynamicGraph[removedEdge[0]].remove(removedEdge[1])
		dynamicGraph[removedEdge[1]].remove(removedEdge[0])

		## Compute the betweenness of the new graph.
		edgeBetweennes = computeBetweennes(graphVertices, dynamicGraph)

	return optCommunitySet


# Task 2.1
edgeBetweennes = computeBetweennes(graphVertices, adjcacencyList)

## Log File.
logFile = './logFile.txt'

## Task 2.2
optCommunitySet = optCommunities(graphVertices, adjcacencyList, numEdges, logFile)
optCommunitySet = [list(i) for i in optCommunitySet]
optCommunitySet.sort(key = lambda x: (len(x), x[0]))

with open(outfilePath1, "w") as f:
	for i in edgeBetweennes:
		f.write(f"{i[0]}, {i[1]}\n")

with open(outfilePath2, "w") as f:
	for i in optCommunitySet:
		f.write("'" + "', '".join(i) + "'\n")

tockTime = time.time()
print(tockTime - ticTime)