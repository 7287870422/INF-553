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
sc = SparkContext('local[*]', 'Task1')
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
		hashVal = (((hashParams['a'][i]*sInt + hashParams['b'][i])%hashParams['p'][i])%69997)
		outputList.append(hashVal)

	return outputList

## Defining the number of hash-functions.	
numHashFunctions = 70

## Obtaining the hashing parameters.
hashParams = hashFunction(numHashFunctions, 69997)
## pprint(hashParams)

## Initialise the filter array.
filterArray = [0] * 69997

## Initialise a set for storing the visited userIDs.
visitedIDs = set()

## Initialise the values for fprNum and tnrNum.
fprNum = 0.0
tnrNum = 0.0

## Variable to keep track of the batchIndex.
batchCounter = 0

## Function to perform the bloom-filtering.
def performBloomFiltering(userStream):

	## Globalize variables.
	global filterArray
	global visitedIDs
	global fprNum
	global tnrNum
	global hashParams
	global numHashFunctions
	global batchCounter

	## Iterate over each userValue.
	for userName in userStream:

		## Obtain the hashed values for the current string.
		currHashValues = myhashs(userName)
		
		## Check how many of the hash-values are already set.
		numSetVals = 0
		for i in range(numHashFunctions):
			if (filterArray[currHashValues[i]] == 1):
				numSetVals += 1

		## Computing the number of false-positives and true-negatives.
		if userName not in visitedIDs:
			if (numSetVals == numHashFunctions):
				fprNum += 1.0
			else:
				tnrNum += 1.0	

		## Set the bit in the bitArray for the hashValues.
		for i in range(numHashFunctions):
			filterArray[currHashValues[i]] = 1

		## Add this to the visited set.
		visitedIDs.add(userName)

	## Compute the false positive rate.
	fprRate = fprNum / (fprNum + tnrNum)

	with open(outfilePath, "a") as f:
		f.write(str(batchCounter) + ',' + str(fprRate) + '\n')

	## Update the batchCounter.
	batchCounter += 1

with open(outfilePath, "w") as f:
	f.write('Time,FPR' + '\n')

## Loop.
for i in range(0, numAsks):

	## Obtain the userStream.
	userStream = bxInstance.ask(dataSetPath, streamSize)

	## Perform bloom-filtering.
	performBloomFiltering(userStream)