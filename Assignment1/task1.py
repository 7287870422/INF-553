import os
import sys
import json
import time
import pyspark
from operator import add
from pprint import pprint
from pyspark import SparkContext

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task1')
sc.setLogLevel("ERROR")

## For better performance, we try to assure that data from a unique business user in is one partition.
def partitionByFunction(id):
    return hash(id)

## Defining the path to the data.
dataSetPath = sys.argv[1]

## Output file path.
outfilePath = sys.argv[2]

## Loading the raw data.
rawData = sc.textFile(dataSetPath)

## Loading the data in json format.
jsonData = rawData.map(lambda f: json.loads(f)).map(lambda f : (f['user_id'], f['business_id'], f['date'])).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

# defaultPartitions = jsonData.getNumPartitions()

## Task A.
## Computing the number of entries in the json file.
totalEntries = jsonData.count()

## Task B.
## Computing the number of entries of 2018 in the json file.
num2018Entries = jsonData.filter(lambda f: '2018' in f[2]).count()

## Task C.
## Computing the number of distinct users who wrote reviews.
allUserEntries = jsonData.map(lambda f : (f[0], 1)).partitionBy(None, partitionByFunction).reduceByKey(lambda user, count : user + count)
numUsers = allUserEntries.count()

## Task D.
## Top 10 users who wrote the maximum number of reviews and their number of reviews.
topUsers = allUserEntries.takeOrdered(10, key = lambda f : (-f[1], f[0]))

## Task E.
## Computing the number of distinct business reviews.
allBusiness = jsonData.map(lambda f : (f[1], 1)).partitionBy(None, partitionByFunction).reduceByKey(lambda business, count : business + count)
numBusinessUsers = allBusiness.count()

## Task F.
## Top 10 business who had the maximum number of reviews and their number of reviews.
topBusiness = allBusiness.takeOrdered(10, key = lambda f : (-f[1], f[0]))

## Initialising a dictionary to write data.
dataDict = {
			'n_review' : totalEntries,
			'n_review_2018' : num2018Entries,
			'n_user' : numUsers,
			'top10_user' : topUsers,
			'n_business' : numBusinessUsers,
			'top10_business' : topBusiness
}

with open(outfilePath, 'w') as outfile:
	json.dump(dataDict, outfile)