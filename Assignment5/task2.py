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

## Initialise global tracking variables.
predVals = 0
trueVals = 0

## Variable to keep track of the batchIndex.
batchCounter = 0

def isPrime(num):
	for m in range(2, int(num ** 0.5) + 1):
		if not num % m:
			return False
	return True

def findPrime(num):
	for n in range(num + 1, num + 10000):
		if isPrime(n): return n

## Function to generate the random hash functions for simulating the permutations.
def hashFunction(numHashFunctions, m):

	## Generate random values for a and b.
	a = random.sample(range(1, m), numHashFunctions)
	b = random.sample(range(1, m), numHashFunctions)
	
	## Generate random prime numbers greater than m.
	p = [findPrime(i) for i in random.sample(range(m, m + 10000), numHashFunctions)]

	## Return the hashed value.
	return {'a' : a, 'b' : b, 'p' : p}

## Defining a function to compute the hashed values for a string.
def myhashs(s):

	## Initialise output list.
	outputList = []

	## Convert it to an integer.
	sInt = int(binascii.hexlify(s.encode('utf8')),16)

	## Compute the hash function value for all the hash functions.
	for i in range(numHashFunctions):
		hashVal = (((hashParams['a'][i]*sInt + hashParams['b'][i])%hashParams['p'][i])%600)
		outputList.append(hashVal)

	return outputList

## Defining the number of hash-functions.	
numHashFunctions = 70

## Defining the number of groups.
numGroups = 10

## Defining the number of rows per group.
rowsPerGroup = int(numHashFunctions / numGroups)

## Obtaining the hashing parameters.
hashParams = hashFunction(numHashFunctions, 600)

## Function to run the Flajolet-Martin Algorithm.
def runFlajoletMartin(userStream):

	## Initialise the global variables.
	global numHashFunctions
	global numGroups
	global rowsPerGroup
	global hashParams
	global predVals
	global trueVals
	global batchCounter

	## Initialise a list to store the hashed bins.
	hashedValBins = []

	## Initialise a set to store the unique IDs.
	uniqueIDs = set()

	## Iterate over each userValue.
	for userName in userStream:

		## Obtain the hashed values for the current string.
		currHashValues = myhashs(userName)
		print(currHashValues)
		print(' ')

		## Add this to the unique set.
		uniqueIDs.add(userName)

		## Initialise a list to store the hashed-binary values.
		hashedVals = []

		## Convert the hashed values to a binary number.
		for hashVal in currHashValues:
			binHashVal = bin(hashVal)[2:]
			hashedVals.append(binHashVal)

		## Add to the bin.
		hashedValBins.append(hashedVals)

	## Initialise array for computing the hash-function wise estimates.
	hashFuncEstimates = []

	## Computing the maximum number of trailing zeroes.
	for i in range(0, numHashFunctions):
		maxZeroes = -float('inf')
		for j in range(0, len(hashedValBins)):
			currZeroes = len(hashedValBins[j][i]) - len(hashedValBins[j][i].rstrip("0"))
			maxZeroes = max(maxZeroes, currZeroes)

		## Produce the estimate for the current hash-function.
		hashFuncEstimates.append(2 ** maxZeroes)

	## Group the hash-functions and get the average for each group.
	groupEstimates = []
	for i in range(numGroups):
		sumEstimates = 0.0
		for j in range(rowsPerGroup):
			sumEstimates += hashFuncEstimates[i * rowsPerGroup + j]

		## Compute the average estimate for the current group.
		avgEstimate = sumEstimates / rowsPerGroup
		groupEstimates.append(avgEstimate)

	## Sort the estimates.
	groupEstimates.sort()

	## Take the median.
	medianVal = groupEstimates[int(numGroups / 2)]

	with open(outfilePath, "a") as f:
		f.write(str(batchCounter) + ',' + str(len(uniqueIDs)) + ',' + str(int(medianVal)) + '\n')

	## Update the batchCounter.
	batchCounter += 1

	## Update.
	predVals += medianVal
	trueVals += len(uniqueIDs)

with open(outfilePath, "w") as f:
	f.write('Time,Ground Truth,Estimation' + '\n')

## Loop.
for i in range(0, numAsks):

	## Obtain the userStream.
	userStream = bxInstance.ask(dataSetPath, streamSize)

	## Perform bloom-filtering.
	runFlajoletMartin(userStream)