import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('IRIS.csv')

df.info()

df['species'].value_counts()

df.isnull().sum()

df_values = df.drop(['species'], axis=1)

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

df['species']=le.fit_transform(df['species'])
df.head()

X=df.iloc[:,0:3].values
Y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

print("Accuracy : ", model.score(X_test,Y_test))

Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
c_mat = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix :\n\n", c_mat)

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))