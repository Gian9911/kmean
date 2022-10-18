import csv
import math

import numpy as numpy


results = []

with open("csv-parrotemotion") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
#print(results[144])

final_elem = 130
i = results[0]
j = results[0]
n = 151
x = 0
y = 0
min = 1000000
index = 0
while (final_elem < n):
    n = n -1
    dist = 100
    min = 1000000
    for elem in results:#elem[0] fornisce il nome,elem[1] il pleasureMean e iil 3 l'arousal
        dist=100
        for row in results:
            if row!=elem:
                dist = math.sqrt((float(elem[1])-float(row[1]))*(float(elem[1])-float(row[1]))
                                 + (float(elem[3])-float(row[3]))*(float(elem[3])-float(row[3])))
                if dist < min:
                    if(dist > 0):
                        #print(dist)
                        min = dist
                        i = elem
                        j = row
    if(float(j[1])>float(i[1])):
        i[1] = (float(i[1]) + dist / 2)
    else:
        i[1] = (float(i[1]) - dist / 2)
    if (float(j[3])>float(i[3])):
        i[3] = (float(i[3]) + dist / 2)
    else:
        i[3] = (float(i[3]) - dist / 2)
    i[0] = i[0]+j[0]
    results.remove(j)


for row in results:
    print(row)


