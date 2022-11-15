
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('8 11.csv')
c = []
a = []
pleasure = []
arousal = []
sd = []
for i in range (len(df)):
    c.append(df.loc[i].at['Color'])
    a.append(df.loc[i].at['Alpha'])
    pleasure.append(df.loc[i].at['PleasureMean'])
    arousal.append(df.loc[i].at['ArousalMean'])
    sd.append(max(df.loc[i].at['Pleasuresd'], df.loc[i].at['Arousalsd'])*10000)

plt.scatter(x=pleasure, y=arousal, c=c, alpha=a, s=sd)
plt.show()