import os
import sys
import csv
import json
import time
import math
import random
import pyspark
import numpy as np
import xgboost as xgb
from operator import add
from pprint import pprint
from pyspark import SparkContext
from itertools import combinations, product
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

ticTime = time.time()

## Initialising the SparkContext.
sc = SparkContext('local[*]', 'Task3')
sc.setLogLevel("ERROR")

## Defining the path to the input folder.
folderPath = sys.argv[1]

## Defining the path to the testing data. 
testDataSetPath = sys.argv[2]

## Output file path.
outfilePath = sys.argv[3]

## Reading the training data.
trainDataSetPath = folderPath + '/yelp_train.csv'

## Loading the raw data.
rawDataTrain = sc.textFile(trainDataSetPath)
rawDataTest = sc.textFile(testDataSetPath)

## Filter out lines.
headerTrain = rawDataTrain.first()
headerTest = rawDataTest.first()

## Loading the CSV data.
yelpDataTrain = rawDataTrain.filter(lambda line: line != headerTrain).map(lambda f: f.split(","))
yelpDataTest = rawDataTest.filter(lambda line: line != headerTest).map(lambda f: f.split(","))

## Function for gathering information about the user.
def userFeatures(currEntry):
	return (currEntry[0], (currEntry[1], currEntry[2]))

## Function for gathering information about the business.
def businessFeatures(currEntry):
	return (currEntry[0], (currEntry[1], currEntry[2]))

## Reading the user information file.
userInfoPath = folderPath + '/user.json'
userData = sc.textFile(userInfoPath)

## Loading the data in json format.
jsonDataUser = userData.map(lambda f: json.loads(f)).map(lambda f : ((f['user_id'], (f['review_count'], f['average_stars'])))).collectAsMap()

# ## Reading the business information file.
businessInfoPath = folderPath + '/business.json'
businessData = sc.textFile(businessInfoPath)

## Loading the data in json format.
jsonDataBusiness = businessData.map(lambda f: json.loads(f)).map(lambda f : ((f['business_id'], (f['review_count'], f['stars'])))).collectAsMap()

## Function for preparing the training data for model training.
def prepData(userBusinessPair, jsonDataUser, jsonDataBusiness, testInput = False):

	## Extract the current user and business.
	if (testInput):
		currUser, currBusiness, currRating = userBusinessPair[0], userBusinessPair[1], -1.0
	else:
		currUser, currBusiness, currRating = userBusinessPair[0], userBusinessPair[1], userBusinessPair[2]

	## Case of cold starts.
	if (currUser not in jsonDataUser.keys() or currBusiness not in jsonDataBusiness.keys()):
		return [currUser, currBusiness, None, None, None, None, None]

	## Extract the relevant user information.
	reviewCountU, avgStarsU = jsonDataUser[currUser]

	## Extract the relevant business information.
	reviewCountB, avgStarsB = jsonDataBusiness[currBusiness]
	
	return [currUser, currBusiness, float(reviewCountU), float(avgStarsU), float(reviewCountB), float(avgStarsB), float(currRating)]

## Prepare the training data.
trainData = yelpDataTrain.map(lambda currEntry : prepData(currEntry, jsonDataUser, jsonDataBusiness)).collect()

## Convert the trainData to a numpy matrix.
trainDataMat = np.array(trainData)

# ## Create the feature set and true values.
XTrain, YTrain = trainDataMat[:, 2 : -1], trainDataMat[:, -1]
XTrain, YTrain = np.array(XTrain, dtype = 'float'), np.array(YTrain, dtype = 'float')

## Train the model.
xgbModel = xgb.XGBRegressor(objective = 'reg:linear')
xgbModel.fit(XTrain,YTrain)

## Prepare the testing data.
testData = yelpDataTest.map(lambda currEntry : prepData(currEntry, jsonDataUser, jsonDataBusiness, True)).collect()

## Convert the testData to a numpy matrix.
testDataMat = np.array(testData)

# ## Create the feature set and true values.
XTest, YTest = testDataMat[:, 2 : -1], testDataMat[:, -1]
XTest, YTest = np.array(XTest, dtype = 'float'), np.array(YTest, dtype = 'float')

## Make the predictions.
modelPreds = xgbModel.predict(XTest)

## Concatenate the names and predictions.
predictVals = np.c_[testDataMat[:, : 2], modelPreds]

fout = open(outfilePath, mode = 'w')
fwriter = csv.writer(fout, delimiter = ',', quoting = csv.QUOTE_MINIMAL)
fwriter.writerow(['user_id', ' business_id', ' prediction'])
for pair in predictVals:
    fwriter.writerow([str(pair[0]), str(pair[1]), float(pair[2])])
fout.close()

tockTime = time.time()
print(tockTime - ticTime)