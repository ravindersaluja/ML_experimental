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

# Dummy encoding or One Hot Encoding
df.head()
from sklearn.preprocessing import OneHotEncoder
# del(ohe)
ohe = OneHotEncoder(sparse=False) # Making it dense
ohe.fit_transform(df[['Sex']])
ohe.categories_

ohe.fit_transform(df[['Embarked']])
ohe.categories_

# Dummy encoding or one hot encoding
# ---------------------------------------
# Changing X
X = df.drop('Survived', axis=1)
X.head()

# Column transformer
from sklearn.compose import make_column_transformer

# Treating Pclass as a numeric variable and not as a categorical variable and hence not ohe it.

# Use column transformer when you have features in the df that need different preprocessing
column_trans = make_column_transformer((OneHotEncoder(), ['Sex', 'Embarked']), remainder='passthrough')
column_trans.fit_transform(X)

# Creating pipeline
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, logreg)

# Here, the cross val score will first split the data that is X and y in 5 sets(cv = 5) and then run the 'pipe' i.e. the pipeline
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# Now fit and transform the pipe(instead of model fitting and transforming)
pipe.fit(X,y)

# Creating random test data, as we didn't split the data earlier into train and test
X_new = X.sample(5, random_state=99) 
X_new

# Now predicting the pipe on the new text data
pipe.predict(X_new) #This is y pred of the new data






