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

## Reading csv file
<img width="565" alt="s1" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/ceb38bed-3c14-4baa-b7d5-9b0a5b4628f5">

## Skew
<img width="215" alt="s2" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/a6621e0d-8895-4d91-b847-e646ccdb0328">

## Function Transformation:

## Log Transformation
<img width="345" alt="s3" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/fc805d60-444e-42d2-9685-6df9a879bef9">

## Reciprocal Transformation
<img width="348" alt="s4" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/dafec8b4-0a7c-4f04-bcb1-1ce4e3e6daf3">

## Square root Transformation
<img width="348" alt="s5" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/e5b3840e-b94f-4383-ba54-d1adf7bcaa7a">

## Square Transformation
<img width="344" alt="s6" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/136b949c-e319-40a6-8c4e-250191925fca">

## Power Transformation:

## Boxcox method
<img width="738" alt="s7" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/6417c408-de6e-4875-a235-005fb97dfb9b">

## Yeojohnson method
<img width="858" alt="s8" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/84a4e2b6-3f40-4b22-b60a-8fe6728a9871">

## Quantile Transformation:
<img width="859" alt="s9" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/7f8c7dd8-ea83-4d4d-951d-be0cb6266e9c">

## Plotting Moderately Negative Skew
<img width="439" alt="s10" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/68f1f853-a670-4651-9d28-9e7930227181">

## Plotting Modified Moderately Negative Skew
<img width="432" alt="s11" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/839eb01c-6563-4298-a19e-d21b55efc9ee">

## Plotting Highly Negative Skew
<img width="443" alt="s12" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/2cc3a06f-760b-403b-9fb5-649d6144d893">

## Plotting Modified Highly Negative Skew
<img width="431" alt="s13" src="https://github.com/SmritiManikand/ODD2023-Datascience-Ex06/assets/113674204/d0dddf34-a317-4014-86ea-e2a9010bf20a">


## Result:
Thus feature transformation is done for the given dataset.
