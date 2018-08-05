#Linear Regression
#Using sklearn's linear regression model

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

                                                #read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']].values.tolist()
y_values = dataframe[['Body']].values.tolist()

                                                #train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

wt = int(input())
print("Body", body_reg.predict(wt))             #Predict on custom input

                                                #visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.xlabel('Brain weight', fontsize=18)
plt.ylabel('Body weight', fontsize=16)
plt.show()
