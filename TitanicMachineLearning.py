import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

train = pd.read_csv('titanic_train.csv')
#print(train.head())

#Plotting missing data
'''
#print(train.isnull())
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
plt.show()
'''
sns.set_style('whitegrid')
#Checking Male vs Female
sns.countplot(x='Survived',data=train,hue='Sex')
plt.show()
'''
#Checking survived between classes
sns.countplot(x='Survived',data=train,hue='Pclass')
plt.show()

#Checking the age
sns.distplot(train['Age'].dropna(),kde=False)
plt.show()
#Checking family members
sns.countplot(x='SibSp',data=train)
plt.show()

#Checking the fare
train['Fare'].hist()
plt.show()
'''
#Age vs Class
sns.boxplot(x='Pclass',y='Age',data=train)
plt.show()

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if(Pclass == 1):
            return 37
        elif Pclass == 2:
            return 29
        else :
            return 24
    else :
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

#No missing values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
plt.show()

sex = pd.get_dummies(data=train['Sex'],drop_first=True)
embark = pd.get_dummies(data=train['Embarked'],drop_first=True)


train = pd.concat([train,sex,embark],axis=1)

#Dropping useless colomuns
train.drop(['Sex' ,'Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
print(train.head(2))


#Splitting train in test train for practise

X = train.drop('Survived',axis=1)
y = train['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(train.head())

s = input("Enter the Pclass,Age,SibSp,Parch,fare,Male,Q,S")
numbers = list(map(int, s.split()))

print(logmodel.predict([numbers]))
