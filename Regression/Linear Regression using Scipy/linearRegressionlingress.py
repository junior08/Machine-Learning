#Linear Regression
#Using scipy.stats linear regression model linregress

import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy import stats
import matplotlib.pyplot as plt

                                                #read data from brain body file
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']].values.tolist()
y_values = dataframe[['Body']].values.tolist() #segregating columns

x = np.array(x_values)
                                                
slope, intercept, r_value, p_value, std_err, *a = stats.mstats.linregress(x_values, y_values)

wt = int(input())
print("Body", intercept + slope * wt)             #Predict on custom input

                                                #visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, intercept +slope * x)
plt.xlabel('Brain weight', fontsize=18)
plt.ylabel('Body weight', fontsize=16)
plt.show()#plots 
