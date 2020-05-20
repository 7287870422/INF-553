import os
import sys
import csv
import json
import time
import random
import pyspark
import binascii
from operator import add
from pprint import pprint
from blackbox import BlackBox
from pyspark import SparkContext
from itertools import combinations

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task2')
sc.setLogLevel("ERROR")

## Set the random seed.
random.seed(553)

## Defining the path to the data. 
dataSetPath = sys.argv[1]

## Defining the stream size.
streamSize = int(sys.argv[2])

## Defining the number of asks.
numAsks = int(sys.argv[3])

## Defining the path to the output file.
outfilePath = sys.argv[4]

## Initiate an instance of blackbox.
bxInstance = BlackBox()

## Initiating a list to hold the user ID's.
userIDList = []

## Defining the maximum size of the list.
maxListSize = 100

## Initialising  a variable to hold the number 
## of elements seen till now.
numElems = 0

## Function to run the reservoir sampling.
def reservoirSampling(userStream):

	## Global parameters.
	global userIDList
	global maxListSize
	global numElems

	for userName in userStream:

		## Update the number of elements.
		numElems += 1

		## Check it can be added to the list.
		if (len(userIDList) < maxListSize):
			userIDList.append(userName)

		## Otherwise, check whether we want to replace the current userID with an old one
		## or discard it altogether.
		else:	

			## Compute the keeping probability.
			probVal = (random.randint(0, 100000) % numElems)
			
			## If we decide to keep the current ID.
			if (probVal < 100):

				## Compute the index to be replaced.
				replaceIdx = random.randint(0,100000) % 100
			
				## Add the current element to the list.
				userIDList[replaceIdx] = userName

	## Print Stats.
	if (numElems != 0 and numElems % 100 == 0):
		print(numElems, userIDList[0], userIDList[20], userIDList[40], userIDList[60], userIDList[80])

		with open(outfilePath, "a") as f:
			f.write(str(numElems) + ',' + str(userIDList[0]) + ',' + str(userIDList[20]) + ',' + str(userIDList[40]) + ',' + str(userIDList[60]) + ',' + str(userIDList[80]) + '\n')


with open(outfilePath, "w") as f:
	f.write('seqnum,0_id,20_id,40_id,60_id,80_id' + '\n')

## Loop.
for i in range(0, numAsks):

	## Obtain the userStream.
	userStream = bxInstance.ask(dataSetPath, streamSize)

	## Perform bloom-filtering.
	reservoirSampling(userStream)