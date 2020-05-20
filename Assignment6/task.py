import os
import sys
import csv
import json
import time
import random
import pyspark
import binascii
import numpy as np
from operator import add
from pprint import pprint
from pyspark import SparkContext
from itertools import combinations
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task1')
sc.setLogLevel("ERROR")

## Defining the path to the data. 
dataSetPath = sys.argv[1]

## Defining the number of clusters.
numClusters = int(sys.argv[2])

## Defining the path to the output file.
outfilePath = sys.argv[3]

## Load the data.
file = open(dataSetPath, "r")
allData = np.array(file.readlines())
file.close()

## Defining a function that takes data points and prepares it in the format suitable for K-Means clustering.
def KMeansCompatible(dataChunk):

	## Create a placeholder array.
	numElems = len(dataChunk)
	rowLen = len(dataChunk[0].strip('\n').split(','))
	usefulDataMat = np.empty(shape = (numElems, rowLen))

	## Convert strings to numpy arrays.
	for i in range(0, numElems):
		usefulDataMat[i] = np.array(dataChunk[i].strip('\n').split(','))

	return usefulDataMat

## Convert all the data into a numpy array.
allDataNP = KMeansCompatible(allData)
groundTruthLabels = allDataNP[:, 1]
groundTruthLabels = groundTruthLabels.tolist()

## Shuffle.
np.random.shuffle(allData)

## Global counter for keeping track of the number of CS clusters.
csClusterNum = 0

# Split the data into 5 partitions.
dataSplits = np.array_split(allData, 5)

## Defining a function to compute the Mahanolobis Distance.
def computeMahanolobis(currPoint, clusterCentroid, clusterSD):

	## Subtract the currPoint coordinates from the clusterCentroid.
	numTerm = np.subtract(currPoint, clusterCentroid)

	## Normalize by the clusterVar.
	normalizedTerm = np.divide(numTerm, clusterSD)

	## Square each element.
	normalizedTerm = np.square(normalizedTerm)

	## Add up the vector column-wise and take the square root.
	mahanolobisDist = np.sqrt(np.sum(normalizedTerm, axis = 0))

	return mahanolobisDist

## Creating a function to compute the mean and variance of a cluster.
def computeStats(dimSum, dimSquaredSum, numPoints):

	## Computing the cluster centroid.
	clusterCentroid = dimSum / numPoints

	## Computing the standard deviation.
	clusterVar = (np.subtract((dimSquaredSum / numPoints), (np.square(clusterCentroid))))
	clusterSD = np.sqrt(clusterVar)

	return clusterCentroid, clusterSD

## Defining a function to generate the DS/CS/RS.
def generateSets(kmeansRun, dataChunk, dsInit, rsInit, csInit):

	## Globalize the dictionaries.
	global dsStatDict
	global csStatDict

	## Globalize the CS Cluster Counter.
	global csClusterNum

	## Initialise a set to store the set of retained points.
	retainedSet = set()

	## Obtain the cluster assignments.
	clusterAssignDict = defaultdict(list)
	clusterAssigns = kmeansRun.labels_
	for i, currAssign in enumerate(clusterAssigns):
		clusterAssignDict[currAssign].append(i)

	for key, value in clusterAssignDict.items():

		## Check for RS.
		## If the cluster contains only 1 point, move it to RS.
		if (rsInit):
			if (len(value) == 1):
				retainedSet.add(value[0])
				continue

		if (dsInit or csInit):

			## Obtain the submatrix corresponding to the current cluster.
			newData = dataChunk[:, 2 : ]
			subMat = newData[value, :]

			## Obtain a submatrix for the indices.
			datIdxVec = dataChunk[:, 0]
			datIdxVec = datIdxVec[value]
			datIdxVec = np.asarray(datIdxVec, dtype = int)
			datIdxVec = datIdxVec.tolist()

			## Computing the sum across each column (dimension).
			dimSum = np.sum(subMat, axis = 0)

			## Computing the squared sum across each column (dimension).
			dimSquaredSum = np.sum(np.square(subMat), axis = 0)

			## Compute the necessary statistics.
			numPoints = len(value)
			clusterCentroid, clusterSD = computeStats(dimSum, dimSquaredSum, numPoints)

			## Store the statistics for the DS Set.
			if (dsInit):
				dsStatDict[key].append(numPoints)
				dsStatDict[key].append(dimSum)
				dsStatDict[key].append(dimSquaredSum)
				dsStatDict[key].append(clusterCentroid)
				dsStatDict[key].append(clusterSD)
				dsStatDict[key].append(datIdxVec)

			## Store the statistics for the CS Set.
			if (csInit):
				csStatDict[csClusterNum].append(numPoints)
				csStatDict[csClusterNum].append(dimSum)
				csStatDict[csClusterNum].append(dimSquaredSum)
				csStatDict[csClusterNum].append(clusterCentroid)
				csStatDict[csClusterNum].append(clusterSD)
				csStatDict[csClusterNum].append(datIdxVec)

				## Update the cluster number.
				csClusterNum += 1

	return retainedSet

## Defining a function to update the DS/CS.
def updateSets(dataPoint, minCluster, dsUpdate, csUpdate):

	## Globalize the dictionaries.
	global dsStatDict
	global csStatDict

	if (dsUpdate):

		## Update the necessary statistics.
		numPoints = dsStatDict[minCluster][0] 
		numPoints += 1
		
		## Computing the sum across each column (dimension).
		dimSum = dsStatDict[minCluster][1]
		dimSum = np.add(dimSum, dataPoint[2:])
		
		## Computing the squared sum across each column (dimension).
		dimSquaredSum = dsStatDict[minCluster][2]
		dimSquaredSum = np.add(dimSquaredSum, np.square(dataPoint[2:]))
		
		## Add the index of the current point to the indices list.
		datIdxVec = dsStatDict[minCluster][5]
		datIdxVec.append(int(dataPoint[0]))

		## Computing the necessary statistics.
		clusterCentroid, clusterSD = computeStats(dimSum, dimSquaredSum, numPoints)

		## Update the statistics for the DS Set.
		dsStatDict[minCluster][0] = numPoints
		dsStatDict[minCluster][1] = dimSum
		dsStatDict[minCluster][2] = dimSquaredSum
		dsStatDict[minCluster][3] = clusterCentroid
		dsStatDict[minCluster][4] = clusterSD
		dsStatDict[minCluster][5] = datIdxVec

	elif (csUpdate):

		## Update the necessary statistics.
		numPoints = csStatDict[minCluster][0] 
		numPoints += 1

		## Computing the sum across each column (dimension).
		dimSum = csStatDict[minCluster][1]
		dimSum = np.add(dimSum, dataPoint[2:])

		## Computing the squared sum across each column (dimension).
		dimSquaredSum = csStatDict[minCluster][2]
		dimSquaredSum = np.add(dimSquaredSum, np.square(dataPoint[2:]))

		## Add the index of the current point to the indices list.
		datIdxVec = csStatDict[minCluster][5]
		datIdxVec.append(int(dataPoint[0]))

		## Computing the necessary statistics.
		clusterCentroid, clusterSD = computeStats(dimSum, dimSquaredSum, numPoints)

		## Update the statistics for the CS Set.
		csStatDict[minCluster][0] = numPoints
		csStatDict[minCluster][1] = dimSum
		csStatDict[minCluster][2] = dimSquaredSum
		csStatDict[minCluster][3] = clusterCentroid
		csStatDict[minCluster][4] = clusterSD
		csStatDict[minCluster][5] = datIdxVec

## Set of all data indices.
allIndices = set()

## Obtaining data for the initial run of K-Means.
initData = KMeansCompatible(dataSplits[0])
initDataRun = initData[:, 2 :]

## Add all the indices to allIndices.
allIndicesChunk = initData[:, 0]
allIndicesChunk = np.asarray(allIndicesChunk, dtype = int)
allIndicesChunk = allIndicesChunk.tolist()
for datIdx in allIndicesChunk:
	allIndices.add((datIdx))

## Run K-Means.
kmeansRun1 = KMeans(n_clusters = numClusters * 5).fit(initDataRun)

## Initialising a set to store the retained set (RS) points.
retainedSetGlob = set()

## Initialise a set to store the points belonging to non-unit length clusters.
nonUnitClusters = set()

## Obtain the cluster assignments.
clusterAssignDict = defaultdict(list)
clusterAssigns = kmeansRun1.labels_
for i, currAssign in enumerate(clusterAssigns):
	clusterAssignDict[currAssign].append(i)

## Obtain the clusters containing only 1 point.
for key, value in clusterAssignDict.items():
	if (len(value) == 1):
		retainedSetGlob.add(int(value[0]))
	else:
		for pointIndex in value:
			nonUnitClusters.add(pointIndex)

## Obtain the submatrix corresponding to the non-RS points.
newInitData = initData[list(nonUnitClusters), :]

# Defining the closeness threshold.
rowLen = initData.shape[1]
closenessThreshold = 2 * np.sqrt(rowLen - 2)

## Run K-Means again on newInitData.
kmeansRun2 = KMeans(n_clusters = numClusters).fit(newInitData[:, 2 : ])

## Initialise a dictionary to store the CS Statistics.
csStatDict = defaultdict(list)

## Initialise a dictionary to store the DS Statistics.
dsStatDict = defaultdict(list)

## Create the DS Set.
generateSets(kmeansRun2, newInitData, True, False, False)

# Access the actual RS points.
rsPoints = initData[list(retainedSetGlob), :]

## Run K-Means on the retained set if size is appropriate.
if (len(retainedSetGlob) >= 5 * numClusters):
	
	## K-means.	
	kmeansRun3 = KMeans(n_clusters = 5 * numClusters).fit(rsPoints[:, 2 : ])

	## Generate the CS and RS.
	retainedSet = generateSets(kmeansRun3, rsPoints, False, True, True)

	## Union the retained sets.
	retainedSetGlob = retainedSet

## Variable for computing the number of discard points.
numDiscard = 0

for key, value in dsStatDict.items():
	numDiscard += value[0]

## Variable for computing the number of compression points.
numCompression = 0

for key, value in csStatDict.items():
	numCompression += value[0]

## Access the actual RS points.
rsPoints = initData[list(retainedSetGlob), :]

## Round 1 Stats.
with open(outfilePath, "w") as f:
	f.write('The intermediate results:' + '\n')
	f.write('Round 1: ' + str(numDiscard) + ',' + str(len(list(csStatDict.keys()))) + ',' + str(numCompression) + ',' + str(len(retainedSetGlob)) + '\n')

## Loop Logic.
for j in range(1, 5):

	## Set to keep track of the data points in the RS for the current iteration.
	currIterRS = set()
	newDataChunk = dataSplits[j]

	## Convert it to a suitable format.
	dataChunkFormatted = KMeansCompatible(dataSplits[j])

	## Add indices to the global set.
	allIndicesChunk = dataChunkFormatted[:, 0]
	allIndicesChunk = np.asarray(allIndicesChunk, dtype = int)
	allIndicesChunk = allIndicesChunk.tolist()
	for datIdx in allIndicesChunk:
		allIndices.add((datIdx))

	for i, dataPoint in enumerate(dataChunkFormatted):

		## Boolean flag to check whether the current data point has been assigned to a cluster.
		pointAssigned = False

		## Compute the Mahanolobis distance from each of the cluster centroid in DS.
		mahanolobisDistVecDS = []
		for key, value in dsStatDict.items():
			mahanolobisDistDS = computeMahanolobis(dataPoint[2: ], value[3], value[4])
			mahanolobisDistVecDS.append((mahanolobisDistDS, key))

		## Sort the array.
		mahanolobisDistVecDS.sort(key = lambda f: f[0])

		## Take the minimum distance.
		minDistDS = mahanolobisDistVecDS[0][0]

		## Check whether this point is to be inserted in the DS.
		if (minDistDS < closenessThreshold):

			## Update boolean flag.
			pointAssigned = True

			## Determine the cluster it belongs to.
			minClusterDS = mahanolobisDistVecDS[0][1]

			## Update the DS.
			updateSets(dataPoint, minClusterDS, True, False)

		## If CS Dict is not empty.
		if (not pointAssigned and len(csStatDict) != 0):
			
			## Compute the Mahanolobis distance from each of the cluster centroid in CS.
			mahanolobisDistVecCS = []
			for key, value in csStatDict.items():
				mahanolobisDistCS = computeMahanolobis(dataPoint[2: ], value[3], value[4])
				mahanolobisDistVecCS.append((mahanolobisDistDS, key))

			## Sort the array.
			mahanolobisDistVecCS.sort(key = lambda f: f[0])

			## Take the minimum distance.
			minDistCS = mahanolobisDistVecCS[0][0]

			## Check whether this point is to be inserted in the CS.
			if (minDistCS < closenessThreshold):

				## Update boolean flag.
				pointAssigned = True

				## Determine the cluster it belongs to.
				minClusterCS = mahanolobisDistVecCS[0][1]

				## Update the CS.
				updateSets(dataPoint, minClusterCS, False, True)

		## If the current point is to be assigned to RS.
		if (not pointAssigned):
			currIterRS.add(i)

	## Obtain the points in the RS.
	currclusteringData = dataChunkFormatted[list(currIterRS), :]

	## Concatenate the two arrays.
	rsPoints = np.concatenate((currclusteringData, rsPoints))

	# Run K-Means on the retained set if size is appropriate.
	if (rsPoints.shape[0] >= numClusters * 5):
		
		## K-means.	
		kmeansRun4 = KMeans(n_clusters = numClusters * 5).fit(rsPoints[:, 2 : ])

		## Generate the CS and RS.
		retainedSet = generateSets(kmeansRun4, rsPoints, False, True, True)

		## Update the RS.
		rsPoints = rsPoints[list(retainedSet), :]

		## List of the clusters in CS.
		csClusterList = list(csStatDict.keys())

		## List to keep track of the clusters that have been merged.
		mergedClusters = []

		## Compute the Mahanolobis Distances of the CS Clusters.
		for k in range(0, len(csClusterList)):

			## Boolean flag to check whether current cluster was merged or not.
			wasMerged = False

			for l in range(k + 1, len(csClusterList)):

				## Obtain the centroid for cluster 1.
				clusterCentroid1 = csStatDict[csClusterList[k]][3]

				## Obtain the centroid and SD for cluster 2.
				clusterSD2 = csStatDict[csClusterList[l]][4]
				clusterCentroid2 = csStatDict[csClusterList[l]][3]

				## Distance.
				mhDist = computeMahanolobis(clusterCentroid1, clusterCentroid2, clusterSD2)

				## Check if these two clusters are to be merged.
				if (mhDist < closenessThreshold):

					## Check whether the current pair will merge to an already existing cluster
					## or will form a new cluster.
					for r in range(0, len(mergedClusters)):
						if ((csClusterList[k] in mergedClusters[r]) or (csClusterList[l] in mergedClusters[r])):
							mergedClusters[r].add(csClusterList[k])
							mergedClusters[r].add(csClusterList[l])

							## Update Boolean Flag.
							wasMerged = True

					## If the current pair forms a new cluster.
					if (not wasMerged):
						newSet = set([csClusterList[k], csClusterList[l]])
						mergedClusters.append(newSet)

						## Update Boolean Flag.
						wasMerged = True

		## Create a list of CS Clusters to be deleted.
		delList = []

		## Merge the CS Clusters.
		for e in range(0, len(mergedClusters)):

			## Obtain the list of CS Clusters that will be merged.
			csClustersMerged = list(mergedClusters[e])

			## Initialise values.
			numPoints = csStatDict[csClustersMerged[0]][0]

			## Computing the sum across each column (dimension).
			dimSum = csStatDict[csClustersMerged[0]][1]

			## Computing the squared sum across each column (dimension).
			dimSquaredSum = csStatDict[csClustersMerged[0]][2]

			## Initialise the index list.
			datIdxVec = csStatDict[csClustersMerged[0]][5]

			## Add cluster label to delList.
			delList.append(csClustersMerged[0])

			## Add values.
			for h in range(1, len(csClustersMerged)):

				## Update the number of points.
				numPoints += csStatDict[csClustersMerged[h]][0]

				## Update Sum.
				dimSum = np.add(dimSum, csStatDict[csClustersMerged[h]][1])

				## Update Sum-Square.
				dimSquaredSum = np.add(dimSquaredSum, csStatDict[csClustersMerged[h]][2])

				## Update the data indices.
				datIdxVec += csStatDict[csClustersMerged[h]][5]

				## Add cluster label to delList.
				delList.append(csClustersMerged[h])

			## Compute the statistics.
			clusterCentroid, clusterSD = computeStats(dimSum, dimSquaredSum, numPoints)

			## Add values to a new cluster.
			csStatDict[csClusterNum].append(numPoints)
			csStatDict[csClusterNum].append(dimSum)
			csStatDict[csClusterNum].append(dimSquaredSum)
			csStatDict[csClusterNum].append(clusterCentroid)
			csStatDict[csClusterNum].append(clusterSD)
			csStatDict[csClusterNum].append(datIdxVec)

			## Update clusterNum.
			csClusterNum += 1

		## Delete all the clusters that were merged.
		for delClus in delList:
			del csStatDict[delClus]

	## Variable for computing the number of discard points.
	numDiscard = 0

	for key, value in dsStatDict.items():
		numDiscard += value[0]

	## Variable for computing the number of compression points.
	numCompression = 0

	for key, value in csStatDict.items():
		numCompression += value[0]

	## Round 1 Stats.
	if (j != 4):
		with open(outfilePath, "a") as f:
			f.write('Round ' + str(j + 1) + ': ' + str(numDiscard) + ',' + str(len(list(csStatDict.keys()))) + ',' + str(numCompression) + ',' + str((rsPoints.shape[0])) + '\n')

# List of the clusters in CS and DS.
dsClusterList = list(dsStatDict.keys())
csClusterList = list(csStatDict.keys())

## Set to keep track of the clusters that have been merged.
mergedClusters = set()

## Compute the Mahanolobis Distances.
for q in range(0, len(csClusterList)):

	## Boolean flag to check whether current CS cluster was merged or not.
	wasMerged = False

	## Compute the Mahanolobis distance from each of the cluster centroid in DS.
	mahanolobisDistVecCSDS = []
	for key, value in dsStatDict.items():

		mahanolobisDistCSDS = computeMahanolobis(csStatDict[csClusterList[q]][3], value[3], value[4])
		mahanolobisDistVecCSDS.append((mahanolobisDistCSDS, key))

	## Sort the array.
	mahanolobisDistVecCSDS.sort(key = lambda f: f[0])

	## Take the minimum distance.
	minDistCSDS = mahanolobisDistVecCSDS[0][0]

	## Check whether this point is to be inserted in the DS.
	if (minDistCSDS < closenessThreshold):

		## Update boolean flag.
		pointAssigned = True

		## Determine the cluster it belongs to.
		minClusterCSDS = mahanolobisDistVecCSDS[0][1]

		## Update the DS Statistics.
		## If this is a consequent merge.
		numPoints = dsStatDict[minClusterCSDS][0] 
		numPoints += csStatDict[csClusterList[q]][0]

		## Computing the sum across each column (dimension).
		dimSum = np.add(dsStatDict[minClusterCSDS][1], csStatDict[csClusterList[q]][1])

		## Computing the squared sum across each column (dimension).
		dimSquaredSum = np.add(dsStatDict[minClusterCSDS][2], csStatDict[csClusterList[q]][2])

		## Index List.
		datIdxVec = dsStatDict[minClusterCSDS][5] + csStatDict[csClusterList[q]][5]

		## Computing the necessary statistics.
		clusterCentroid, clusterSD = computeStats(dimSum, dimSquaredSum, numPoints)

		## Update the statistics for the CS Set.
		dsStatDict[minClusterCSDS][0] = numPoints
		dsStatDict[minClusterCSDS][1] = dimSum
		dsStatDict[minClusterCSDS][2] = dimSquaredSum
		dsStatDict[minClusterCSDS][3] = clusterCentroid
		dsStatDict[minClusterCSDS][4] = clusterSD
		dsStatDict[minClusterCSDS][5] = datIdxVec

		## Delete the current CS Cluster.
		del csStatDict[csClusterList[q]]

## Set of points which are in DS.
dsPoints = set()

## List of tuples containing data index along with cluster assignment.
clusterAssignments = []

for key, value in dsStatDict.items():

	## Access points assigned to this cluster.
	clusPoints = value[5]
	for labPoint in clusPoints:
		clusterAssignments.append((labPoint, key))
		dsPoints.add(labPoint)

## Set of outlier points.
outlierPoints = allIndices - dsPoints

## Assign a cluster label of -1 to outlier points.
for rsPoint in outlierPoints:
	clusterAssignments.append((rsPoint, -1))

## Sort the assignments list.
clusterAssignments.sort(key = lambda f : f[0])

## Predicted Assignments.
predAssign = []

## Variable for computing the number of discard points.
numDiscard = 0

for key, value in dsStatDict.items():
	numDiscard += value[0]

## Variable for computing the number of compression points.
numCompression = 0

for key, value in csStatDict.items():
	numCompression += value[0]

with open(outfilePath, "a") as f:
	f.write('Round ' + str(j + 1) + ': ' + str(numDiscard) + ',' + str(len(list(csStatDict.keys()))) + ',' + str(numCompression) + ',' + str((rsPoints.shape[0])) + '\n')

## Output the results.
with open(outfilePath, "a") as f:
	f.write('\n')
	f.write('The clustering results:' + '\n')

	for i in clusterAssignments:
		f.write(str(i[0]) + ',' + str(i[1]) + '\n')
		predAssign.append(i[1])


## NMI.
nmiVal = normalized_mutual_info_score(groundTruthLabels, predAssign)
print('NMI is  : ', nmiVal)
