import pandas as pd
# Importing the titanic dataframe from kaggle and storing it in df
df = pd.read_csv('http://bit.ly/kaggletrain')

df.shape
# 891 rows and 12 columns

# Check the number of nulls
df.isna().sum()


df.loc[:, ['Survived',  'Pclass', 'Sex', 'Embarked']] #Survived being the target

df.loc[:, ['Survived',  'Pclass', 'Sex', 'Embarked']].isna().sum()

df.loc[df.Embarked.isna()==False, ['Survived',  'Pclass', 'Sex', 'Embarked']]
#Eliminated rows with na in the Embarked column
df.loc[df.Embarked.isna()==False, ['Survived',  'Pclass', 'Sex', 'Embarked']].isna().sum()
 
# Making a new dataframe
df = df.loc[df.Embarked.isna()==False, ['Survived',  'Pclass', 'Sex', 'Embarked']]

df.head()

X = df.loc[:, ['Pclass']]
X.shape #This is 2D and X always has to be 2D

df.loc[:,'Survived'].shape #This on the other hand is 1D
# So,

y = df.Survived
y.shape #1D

# Building a Classification Model using just a single independent variable - Pclass

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.model_selection import cross_val_score

# Cross val with cv 5, and the mean of the score
cross_val_score(logreg, X, y, cv=5, scoring='accuracy').mean()

# Checking the null accuracy
y.value_counts(normalize = True)[0]


df.head()

# Dummy encoding or one hot encoding









