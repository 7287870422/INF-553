import os
import sys
import time
import json
import pyspark
from pprint import pprint
from pyspark import SparkContext

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task2')
sc.setLogLevel("ERROR")

## Defining the path to the data.
dataSetPath = sys.argv[1]

## Output file path.
outfilePath = sys.argv[2]

## Loading the number of partitions for the custom partition.
numInputPartitions = int(sys.argv[3])

## Loading the raw data.
rawData = sc.textFile(dataSetPath)

## Loading the data in json format.
jsonData = rawData.map(lambda f: (json.loads(f)['business_id'], 1)).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

## Computing the number of partitions and number of items in each partition (default).
numPartitionsDefault = jsonData.getNumPartitions()
numItermsPerPartitionDefault = jsonData.glom().map(len).collect()

## Start the timer.
tickDefault = time.time()

## Sorting the data and extracting the top 10 reviews.
sortedBusinessDefault = jsonData.reduceByKey(lambda business, count : business + count).takeOrdered(10, key = lambda f : (-f[1], f[0]))

## End the timer.
tockDefault = time.time()

## For better performance, we try to assure that data from a unique business user in is one partition.
def partitionByBusiness(business_id):
    return hash(business_id) % numInputPartitions

## Top 10 business who had the maximum number of reviews and their number of reviews.
allBusinessCustom = jsonData.partitionBy(numInputPartitions, partitionByBusiness)

## Start the timer.
tickCustom = time.time()

## Reduce by key.
sortedBusinessCustom = allBusinessCustom.reduceByKey(lambda business, count : business + count).takeOrdered(10, key = lambda f : (-f[1], f[0]))

## End the timer.
tockCustom = time.time()

## Computing the number of partitions and number of items in each partition (default).
numPartitionsCustom = allBusinessCustom.getNumPartitions()
numItermsPerPartitionCustom = allBusinessCustom.glom().map(len).collect()

## Initialising a dictionary to write data.
dataDict = {}

## Writing the relevant to the dictionary.
dataDict['default'] = {'n_partition' : numPartitionsDefault,
                       'n_items' : numItermsPerPartitionDefault,
                       'exe_time' : tockDefault - tickDefault}

dataDict['customized'] = {'n_partition' : numPartitionsCustom,
                          'n_items' : numItermsPerPartitionCustom,
                          'exe_time' : tockCustom - tickCustom}

with open(outfilePath, 'w') as outfile:
	json.dump(dataDict, outfile)