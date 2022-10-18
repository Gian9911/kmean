import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import csv


def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Num clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


df = pd.read_csv('new-reduced.csv')
scaler = StandardScaler()

df[['PleasureMean_t','ArousalMean_t']]=scaler.fit_transform(df[['PleasureMean','ArousalMean']])


optimise_k_means(df[['PleasureMean','ArousalMean']],25)#15 for csv-quad1
kmeans = KMeans(n_clusters=18)
kmeans.fit(df[['PleasureMean_t','ArousalMean_t']])



df['kmeans_9']=kmeans.labels_

i = 0
for key, value in df['kmeans_9'].items():
    print(key, ',', df.Name[key] , ',', value)


plt.scatter(x=df['PleasureMean'], y=df['ArousalMean'], c=df['kmeans_9'])
plt.xlim(-1, 1)
plt.ylim(-1,1)
plt.grid(True)
plt.show()

