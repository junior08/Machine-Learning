#Linear Regression
#Using sklearn's linear regression model

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

                                                #read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']].values.tolist()
y_values = dataframe[['Body']].values.tolist()
x_values = sm.add_constant(x_values)

                                                #train model on data
body_reg = sm.OLS(y_values, x_values)
result = body_reg.fit()

coeff = result.params


print("Enter brain weight")
wt = int(input())
ans = coeff[1] * wt + coeff[0]                  #Construct the equation
print("Body weight", ans)                       #Predict on custom input
