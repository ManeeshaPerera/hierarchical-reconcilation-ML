import pandas as pd
from sklearn.cluster import KMeans

dataset = 'wikipedia'

ts_features = pd.read_csv(f'input_data/ts_features/{dataset}.csv')
data = pd.read_csv(f'input_data/{dataset}.csv')[['Level', 'Description']]
array_features = ts_features.values
kmeans = KMeans(n_clusters=20, random_state=0).fit(array_features)


data['cluster'] = kmeans.labels_
data.to_csv(f'input_data/ts_features/{dataset}_clusters.csv', index=False)