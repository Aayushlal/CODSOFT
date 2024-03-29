import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('advertising.csv')

df.shape

df.info()

plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['TV'],y=df['Sales'])
plt.show()

plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Radio'],y=df['Sales'])
plt.show()

plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Newspaper'],y=df['Sales'])
plt.show()

x = df.drop('Sales',axis=1)

x

y = df['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#fitting the model to the dataset
model.fit(X_train, y_train)

#predictions
y_predictions = model.predict(X_test)

y_predictions

# Lets evaluate the model for its accuracy using various metrics such as RMSE and R-Squared
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_predictions, y_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_predictions, y_test)))
print('R-Squared:', metrics.r2_score(y_predictions, y_test))