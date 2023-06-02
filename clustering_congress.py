import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


data = pd.read_csv("114_congress.csv", delimiter=',')
votes = data.copy()

col_num = votes.select_dtypes(include='number').columns


### Sénateurs dans chaque parti.
for i in range(votes['party'].value_counts().shape[0]) :
    votes['party'].value_counts().index[i]
    votes['party'].value_counts()[i]

### Proportion votes positifs pour chaque projet
votes.select_dtypes(include='number').mean()
#Alternative votes.mean(numeric_only=True)

### Nombre votes positifs pour chaque projet
votes.select_dtypes(include='number').mean()*votes.shape[0]


### Distance entre ligne 1 et 3
distance.euclidean(votes.loc[0,col_num], votes.loc[2,col_num])

### Choix aléatoire de deux sénateurs
centroids = votes.iloc[np.random.choice(100,2,replace=False), :]


##### Clustering via Scikit learn
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=0)

senator_distances = kmeans_model.fit_transform(votes[col_num])

##### Explorer les clusters
labels = kmeans_model.labels_

tableau = pd.crosstab(labels, votes['party'])
tableau.reset_index(drop=True, inplace=True)

#####
dem_outliers_index = list()

for i in range(labels.shape[0]) :
    if labels[i] == 1 and votes['party'][i] == 'D' :
        dem_outliers_index.append(i)

democratic_outliers = votes.loc[dem_outliers_index]


plt.scatter(x = senator_distances[: , 0],
            y = senator_distances[: , 1],
            c = labels)
plt.show()
