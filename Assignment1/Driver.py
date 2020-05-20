import os
import time

grandStart = time.time()

os.system('export PYSPARK_PYTHON=python3.6')
os.system('spark-submit task1.py yelpDataset/review.json result1.json')
os.system('spark-submit task2.py yelpDataset/review.json result2.json 30')
os.system('spark-submit task3.py yelpDataset/review.json yelpDataset/business.json result3_1.txt result3_2.json')


grandEnd = time.time()

print("**************************************************************\n\n")

print(grandEnd - grandStart)

print("**************************************************************\n\n")