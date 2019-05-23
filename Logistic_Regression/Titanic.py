import pandas as pd #for data analysis
import numpy as np #for scientific calculaiions
import seaborn as sns #for statistocal plotting
import matplotlib.pyplot as plt
import math

titanic_data= pd.read_csv('/home/kisore/Downloads/logistic regression/train.csv')
titanic_data.head(10)
print('Number of passengers '+str(len(titanic_data.index)))

sns.countplot(x='Survived', data= titanic_data)
sns.countplot(x='Survived',hue='Sex', data= titanic_data)
sns.countplot(x='Survived',hue='Pclass', data= titanic_data)

titanic_data['Age'].plot.hist()

#to get the columns info of dataset
titanic_data.info()

#checkng for missing data
#True->value is null
#False->value is not null
titanic_data.isnull()

titanic_data.isnull().sum()

#to vsualize the missing data
sns.heatmap(titanic_data.isnull())

#mapping class and age of people 
sns.boxplot(x="Pclass",y="Age",data= titanic_data)

#since "survived" is categorical logistic regression can be applied to it

titanic_data.drop("Cabin",axis=1)

#to remove all the null values
titanic_data.dropna(inplace=True)

titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull())

sex=pd.get_dummies(titanic_data['Sex'], drop_first=True)
sex.head(5)


embark=pd.get_dummies(titanic_data['Embarked'], drop_first=True)
embark.head(5)

pcl=pd.get_dummies(titanic_data['Pclass'], drop_first=True)
pcl.head(5)

#concatenaing these data to titanic_data
titanic_data= pd.concat([titanic_data, sex, embark, pcl], axis=1)
titanic_data.head(10)

titanic_data.drop(['Pclass','Sex','PassengerId','Name','Embarked','Ticket'],axis=1, inplace= True)
titanic_data.head(10)

#training data

X=titanic_data.drop('Survived', axis= 1) #indicates X takes all the columns except Survived since this column is our dependent variable
y=titanic_data['Survived']



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train, y_train)

predictions= logmodel.predict(X_test)


from sklearn.metrics import classification_report 
classification_report(y_test, predictions)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

#getting the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

