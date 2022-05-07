import csv
import json
import pandas as pd
import os

yeardict = dict()
attributes = []
count =0
csvdata = pd.read_csv("/Users/jonathancecil/Downloads/01_District_wise_crimes_committed_IPC_2001_2014.csv")
districts = csvdata["DISTRICT"]
years = csvdata["YEAR"].unique()
for i in csvdata:
    if(count>32):
        break
    attributes.append(i)
    count+=1
attributes.pop(0)
attributes.pop(0)
print(years)
print(attributes)

js = json.loads(csvdata.to_json(orient='records'))
# print(json.loads(js))
for year in years:
    y = str(year)
    yeardict[y] = dict()
    for district in districts:
        yeardict[y][district] = dict()
        for attribute in attributes:
            filtered_result = list(filter(lambda x: x['YEAR'] == year and x['DISTRICT'] == district, js))[0]
            yeardict[y][district][attribute] = filtered_result.get(attribute, 0)
            # print(attribute, " : ", filtered_result.get(attribute, ''))

        with open('history.json', 'w') as f:
            f.write(json.dumps(yeardict, indent=4))