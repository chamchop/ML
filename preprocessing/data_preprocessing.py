import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('House_Price.csv', header=0)
p = df.head()
print(p)

percent = np.percentile(df.n_hot_rooms, [99])[0]
print(percent)

uv = df[(df.n_hot_rooms>percent)]
print(uv)

rf = np.percentile(df.rainfall, [1])[0]
print(rf)

jp = sns.jointplot(x="crime_rate", y="price", data=df)
print(jp)

print(df.describe())
print(p.describe())

hb = df.n_hos_beds
hb = df.n_hos_beds.fillna(df.n_hos_beds.mean())
print(df.n_hos_beds)

ad = df['avg-dist'] = (df.dist1+df.dist2)/2
print(ad.describe())

du = pd.get_dummies(df)
print(du)

h = df.head()
cr = h.corr
print(cr)

print(df.head())