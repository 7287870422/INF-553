import os
import sys
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell"

import time
from pprint import pprint
from functools import reduce
from pyspark import SparkContext
from graphframes import GraphFrame
from pyspark.sql import SQLContext, Row

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task4')
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

## Defining the path to the data. 
dataSetPath = sys.argv[1]

## Output file path.
outfilePath = sys.argv[2]

## Loading the raw data.
rawData = sc.textFile(dataSetPath).map(lambda f: f.split(" "))

## Create the mapping.
graphVertices = (rawData.map(lambda f : (f[0])) + rawData.map(lambda f : (f[1]))).distinct().map(lambda x: Row(id = x)).collect()
pprint(graphVertices)
graphEdges = rawData.flatMap(lambda f : ((f[0], f[1]), (f[1], f[0]))).collect()

## Create the vertices.
vertices = sqlContext.createDataFrame(graphVertices, ['id'])

## Create the edges.
edges = sqlContext.createDataFrame(graphEdges, ["src", "dst"])

## Construct the graph.
g = GraphFrame(vertices, edges)

## Run the label propagation algorithm.
result = g.labelPropagation(maxIter = 5).collect()
resultDict = {}

for i in result:
    label = i["label"]
    if label not in resultDict.keys():
        resultDict[label] = []
    resultDict[label].append(str(i["id"]))

ordered_keys = sorted(resultDict, key=lambda k: (len(resultDict[k]), min(resultDict[k])))
with open(outfilePath, "w") as f:
    for key in ordered_keys:
        temp = sorted(resultDict[key])
        f.write("'" + "', '".join(temp) + "'\n")

tockTime = time.time()
print(tockTime - ticTime)