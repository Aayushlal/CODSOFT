import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('Titanic-Dataset.csv')

df.head()

df.shape

df.info()

df.isnull().sum()

df= df.drop(columns='Cabin', axis=1)

df['Age'].fillna(df['Age'].mean(), inplace=True)

print(df['Embarked'].mode())

print(df['Embarked'].mode()[0])

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.isnull().sum()

df.describe()

df['Survived'].value_counts()

sns.set()
sns.countplot(x = 'Survived', data = df)

df['Sex'].value_counts()

sns.countplot(x = 'Sex', data=df)

sns.countplot(x = 'Sex', hue='Survived', data=df)

df['Sex'].value_counts()

df['Embarked'].value_counts()

df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
df.head()

X = df.drop(columns = ['PassengerId','Name','Ticket','Survived','Fare'],axis=1)
Y = df['Survived']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=2)

print(X_train)

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)