import math
import pandas as pd
import os
import csv


def distance(pleasure1, pleasure2, arousal1, arousal2):
    return math.sqrt((pleasure1 - pleasure2) ** 2 + (arousal1 - arousal2) ** 2)


df_label = pd.read_csv('labels.csv')
df_omg = pd.read_csv('omg_TestVideos_WithLabels.csv')
df = pd.read_csv('4 11.csv')

with open('mycsv.csv', mode='w') as mycsv:
    my_writer = csv.writer(mycsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    distanz = 10000000
    for i in range(len(df_omg)):
        emoz = df_omg.loc[i].at['EmotionMaxVote']

        for j in range(len(df)):
            if emoz == df.loc[j].at['Numero']:
                d2 = distance(df_omg.loc[i].at['valence'], df.loc[j].at['PleasureMean'], df_omg.loc[i].at['arousal'],
                              df.loc[j].at['ArousalMean'])
                #print(d2)
                if d2 < distanz:
                    #print('+ piccolo con emozione')
                    distanz = d2
                    new_emoz = df.loc[j].at['Emozioni']
                    #print(new_emoz)
        my_writer.writerow([df_omg.loc[i].at['link'], df_omg.loc[i].at['arousal'], df_omg.loc[i].at['valence'], new_emoz])
        distanz = 100000000
        #print('vado alla prox')
        emoz=''
        new_emoz=''

