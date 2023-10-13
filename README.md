# Ex-06 FEATURE TRANSFORMATION

## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1: Read the given Data.
Step2: Clean the Data Set using Data Cleaning Process.
Step3: Apply Feature Transformation techniques to all the features of the data set.
Step4: Print the transformed features.

## Program:

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Positive Skew"])

df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Moderate Negative Skew_1"],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

## Output:




## Result:
Thus feature transformation is done for the given dataset.
