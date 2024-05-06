import statsmodels.api as sn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('House_Price.csv', header=0)
head = df.head()

X = sn.add_constant(df['room_num'])
print(X)

lm = sn.OLS(df['price'], X).fit()
print(lm.summary())

y = df['price']
print(y)

x = df['room_num']
print(x)

lm2 = LinearRegression()
lm2.fit(X, y)
print(lm2.intercept_, lm2.coef_)

pr = lm2.predict(X)
print(pr)