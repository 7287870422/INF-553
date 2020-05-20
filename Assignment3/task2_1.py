import os
import sys
import csv
import json
import time
import math
import random
import pyspark
import numpy as np
from operator import add
from pprint import pprint
from pyspark import SparkContext
from itertools import combinations, product

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task2')
sc.setLogLevel("ERROR")

## Defining the path to the training data. 
trainDataSetPath = sys.argv[1]

## Defining the path to the testing data. 
testDataSetPath = sys.argv[2]

## Output file path.
outfilePath = sys.argv[3]

## Loading the raw data.
rawDataTrain = sc.textFile(trainDataSetPath)
rawDataTest = sc.textFile(testDataSetPath)

## Filter out lines.
headerTrain = rawDataTrain.first()
headerTest = rawDataTest.first()

## Loading the CSV data.
yelpDataTrain = rawDataTrain.filter(lambda line: line != headerTrain).map(lambda f: f.split(","))
yelpDataTest = rawDataTest.filter(lambda line: line != headerTest).map(lambda f: f.split(","))

## Create a mapping of the form: {ui : [bi, ... , bk]}.
userBusinessMap = yelpDataTrain.map(lambda f : (f[0], f[1])).groupByKey().mapValues(set)

## Collect the userBusinessMap RDD.
userBusinessMapDict = {i: j for i, j in userBusinessMap.collect()}

## Create a mapping of the form: {bi : [ui, ... , uk]}.
businessUserMap = yelpDataTrain.map(lambda f : (f[1], f[0])).groupByKey().mapValues(set)

## Collect the businessUser RDD.
businessUserMapDict = {i: j for i, j in businessUserMap.collect()}

## Create a dictionary with key as (userId, businessId) and value as (ratingVal).
userBusinessRateDict = yelpDataTrain.map(lambda f : ((f[0], f[1]), float(f[2])))

## Collect the userBusinessRateDict RDD.
userBusinessRateDict = {i: j for i, j in userBusinessRateDict.collect()}

## Compute the average rating business-wise.
businessAverage = yelpDataTrain.map(lambda f : (f[1], (float(f[2])))).combineByKey(lambda val: (val,1), lambda x,val : (x[0] + val, x[1] + 1), lambda x,y: (x[0] + y[0], x[1] + y[1] )).mapValues(lambda x: x[0]/x[1])

## Collect the businessAverage RDD.
businessAverageDict = {i: j for i, j in businessAverage.collect()}

## Function to compute the pearson correlation.
def computePredictions(userBusinessPair, userBusinessMapDict, businessUserMapDict, userBusinessRateDict, numNeighbours):

	## Extract the current user and business.
	currUser, currBusiness = userBusinessPair[0], userBusinessPair[1]

	## Cold Start for Business.
	if (currBusiness not in businessUserMapDict.keys()):
		return (currUser, currBusiness, 3.0)

	## Cold Start for User.
	if (currUser not in userBusinessMapDict.keys()):
		return (currUser, currBusiness, 3.0)
		
	## Find out the set of co-rated businesses by the user.
	coRatedBusinessList = userBusinessMapDict[currUser]

	## Initialise a list to hold the pearson correlation.
	pearsonList = []

	## Compute the pearson correlation for the current businessID.
	for coRatedBusiness in coRatedBusinessList:

		## Extract the rating with the coordinate (currUser, coRatedBusiness).
		ratingCellVal = userBusinessRateDict[(currUser, coRatedBusiness)]

		## Compute the set of co-rated users.
		coRatedUsers = businessUserMapDict[currBusiness].intersection(businessUserMapDict[coRatedBusiness])

		if (len(coRatedUsers) == 0 or len(coRatedUsers) == 1):

			## Check the difference between the average ratings of the businesses.
			absDiff = abs(businessAverageDict[currBusiness] - businessAverageDict[coRatedBusiness])
			
			## Create bins.
			if (0 <= absDiff <= 1):

				## Set pearson correlation to be 1.
				pearsonCorrelation = 1.0

				## Add the value to the list.
				pearsonList.append([pearsonCorrelation, pearsonCorrelation * ratingCellVal, abs(pearsonCorrelation)])
				continue

			elif (1 < absDiff <= 2):
				
				## Set pearson correlation to be 0.5.
				pearsonCorrelation = 0.5

				## Add the value to the list.
				pearsonList.append([pearsonCorrelation, pearsonCorrelation * ratingCellVal, abs(pearsonCorrelation)])
				continue				

			else:
				pearsonList.append([0.0, 0.0, 0.0])	
				continue

		## Initialise two vectors to hold the ratings for the two businesses.
		riVec = []
		rjVec = []

		## Fill the vector to hold the rating entries.
		for users in coRatedUsers:

			## Add the ratings to the vector.
			riVec.append(userBusinessRateDict[(users, currBusiness)])
			rjVec.append(userBusinessRateDict[(users, coRatedBusiness)])

		riVec = np.asarray(riVec, dtype = np.float32)
		rjVec = np.asarray(rjVec, dtype = np.float32)

		riNormalized = riVec - businessAverageDict[currBusiness]
		rjNormalized = rjVec - businessAverageDict[coRatedBusiness]

		## Compute the pearson correlation.
		numTerm = np.sum(np.multiply(riNormalized, rjNormalized))
		denTerm = np.sqrt(np.sum(riNormalized ** 2)) * np.sqrt(np.sum(rjNormalized ** 2))

		## If pearson value does not exists.
		if (numTerm == 0 or denTerm == 0):
			pearsonList.append([0.0, 0.0, 0.0])	
			continue

		pearsonCorrelation = numTerm/denTerm

		## If correlation is negative.
		if (pearsonCorrelation < 0):
			continue			

		pearsonList.append([pearsonCorrelation, pearsonCorrelation * ratingCellVal, abs(pearsonCorrelation)])

	# Debug.
	if (len(pearsonList) == 0):
		return (currUser, currBusiness, 3.0)	

	## Sort the list based on pearson correlation value.
	pearsonList = sorted(pearsonList, key = lambda x : -x[0])
	pearsonList = pearsonList[ : numNeighbours]

	## Convert the list to a numpy array.
	pearsonMat = np.array(pearsonList)

	## Sum across columns.
	summedVals = pearsonMat.sum(axis = 0)

	## Case where the currPred is 0 or Nan.
	if (summedVals[1] == 0.0 or summedVals[2] == 0.0):
		return (currUser, currBusiness, 3.0)			

	## Compute the prediction for the current tuple.
	currPred = summedVals[1] / summedVals[2]

	return (currUser, currBusiness, currPred)

## Defining the number of neighbours.
numNeighbours = 15

## Compute the pearson correlation for the required entries.
predictVals = yelpDataTest.map(lambda currEntry : computePredictions(currEntry, userBusinessMapDict, businessUserMapDict, userBusinessRateDict, numNeighbours)).collect()

fout = open(outfilePath, mode = 'w')
fwriter = csv.writer(fout, delimiter = ',', quoting = csv.QUOTE_MINIMAL)
fwriter.writerow(['user_id', ' business_id', ' prediction'])
for pair in predictVals:
    fwriter.writerow([str(pair[0]), str(pair[1]), pair[2]])
fout.close()

tockTime = time.time()
print(tockTime - ticTime)