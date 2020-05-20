import os
import sys
import time
import json
import pyspark
from pprint import pprint
from pyspark import SparkContext

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task3')
sc.setLogLevel("ERROR")

## Defining the path to the data.
dataSetPath2 = sys.argv[1]
dataSetPath1 = sys.argv[2]

## Loading the raw data.
rawData1 = sc.textFile(dataSetPath1)
rawData2 = sc.textFile(dataSetPath2)

## Partition function.
def partitionByCity(city):
    return hash(city)

## Perform operations for Python.

## Start Timer.
tickPython = time.time()

# ## Loading the data in json format.
cityBusinessMap = rawData1.map(json.loads).map(lambda f : (f['business_id'], f['city']))
businessStarMap = rawData2.map(json.loads).map(lambda f : (f['business_id'], f['stars']))

aggregatedData = cityBusinessMap.join(businessStarMap).map(lambda f : (f[1][0], f[1][1])).partitionBy(None, partitionByCity).combineByKey(lambda val: (val,1), lambda x,val : (x[0] + val, x[1] + 1), lambda x,y: (x[0] + y[0], x[1] + y[1] )).mapValues(lambda x: x[0]/x[1]).collect()

## Sort the data in native python.
pythonSortedData = sorted(aggregatedData, key = lambda f: (-f[1], f[0]))
pythonSortedDataFirst10 = pythonSortedData[:10]

## End Timer.
tockPython = time.time()

## Print stats.
print('Python Sorting !')
for i in range(0, len(pythonSortedDataFirst10)):
	print(pythonSortedDataFirst10[i][0])

# Perform operations for PySpark.

## Start Timer.
tickPySpark = time.time()

## Loading the data in json format.
cityBusinessMap = rawData1.map(json.loads).map(lambda f : (f['business_id'], f['city']))
businessStarMap = rawData2.map(json.loads).map(lambda f : (f['business_id'], f['stars']))

sparkSortedData = cityBusinessMap.join(businessStarMap).map(lambda f : (f[1][0], f[1][1])).partitionBy(None, partitionByCity).combineByKey(lambda val: (val,1), lambda x,val : (x[0] + val, x[1] + 1), lambda x,y: (x[0] + y[0], x[1] + y[1] )).mapValues(lambda x: x[0]/x[1]).takeOrdered(10, lambda f : (-f[1], f[0]))

## End Timer.
tockPySpark = time.time()

## Print stats.
print('PySpark Sorting !')
for i in range(0, len(sparkSortedData)):
	print(sparkSortedData[i][0])

## Output file path for part A.
outfilePathA = sys.argv[3]

with open(outfilePathA, 'w') as f:
    f.write("city,stars\n")
    for item in pythonSortedData:
        f.write(item[0] + "," + str(item[1]) + "\n")

## Storing the times.
dataDict = {'m1' : tockPython - tickPython, 'm2' : tockPySpark - tickPySpark}

## Output file path for part B.
outfilePathB = sys.argv[4]

with open(outfilePathB, 'w') as outfile:
  json.dump(dataDict, outfile)