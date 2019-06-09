#Notes: Adding Fare to the logistic regression did not help
#Version 1.2 - Changed how names were processed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("C:\\Users\\Joseph Streb\\Documents\\Python Scripts\\Datasets\\titanic_train.csv")
test_data = pd.read_csv("C:\\Users\\Joseph Streb\\Documents\\Python Scripts\\Datasets\\titanic_test.csv")

#-------------------------------------------------------------------------------------------------------------------
#Data Exploration Techniques
#-------------------------------------------------------------------------------------------------------------------
#Creating tables comparing the independent variable with a categorical variable
#sex_pivot = train_data.pivot_table(index="Sex",values="Survived") #Creates small pivot w/ details about sex and survival
#sex_pivot.plot.bar() #dispays chart created above
#pclass_pivot = train_data.pivot_table(index="Pclass",values="Survived")#Creates small pivot w/ details about class and survival
#pclass_pivot.plot.bar()#dispays chart created above
#fare_pivot = train_data.pivot_table(index="Pclass",values="Fare")#Creates small pivot w/ details about class and survival

#class1=train_data[train_data["Pclass"]==1]
#class2=train_data[train_data["Pclass"]==2]
#class3=train_data[train_data["Pclass"]==3]
#class1["Fare"].plot.hist(alpha=0.5, color = 'red', bins=300)
#class2["Fare"].plot.hist(alpha=0.5, color = 'blue', bins=50)
#lass3["Fare"].plot.hist(alpha=0.5, color = 'yellow', bins=50)
#lt.legend(['class1', 'class2', 'class3'])
#plt.ylim([0,60])
#plt.show()

#Describing data
#train_data['Age'].describe() #gives summary stats on column

#Creating histograms to compare each dependent binary classification
#survived = train_data[train_data["Survived"]==1]
#died = train_data[train_data["Survived"]==0]

#survived["Age"].plot.hist(alpha=0.5, color = 'red', bins=50)
#died["Age"].plot.hist(alpha=0.5, color = 'blue', bins=50)

#plt.legend(['survived', 'died'])

#lt.show()

#Exploring the Name Data
#train_data1 = train_data
#train_data1["Name"] = train_data['Name'].str.replace('[^\w\s]','')
#words = pd.Series(' '.join(train_data1['Name']).lower().split()).value_counts()[90:120]

#-------------------------------------------------------------------------------------------------------------------
#Feature Engineering
#-------------------------------------------------------------------------------------------------------------------
#Feature 1: Age
#Comment: Oppurtunity for model improvement in changing the location of the cuts?
def process_age(df, cut_points, label_names):
    df["Age"]=df["Age"].fillna(-0.5)
    df['Age_categories']= pd.cut(df['Age'],cut_points, labels = label_names)
    return df

def process_titles(df, name_col, titles):
    df[name_col] = df[name_col].str.lower().replace('[^\w\s]','')
    df[name_col+'_titles'] = df[name_col].str.contains('|'.join(titles))*1
    return df

#Processing Age Inputs
age_cut_points = [-1,0,5,12,18,35,60,100]
age_label_names = ['Missing', 'Infant', 'Child','Teenager','YoungAdult', 'Adult', 'Senior']

#Process Name Inputs
name_col = 'Name'
name_titles = ['sir','dr','sr','jr','master','rev']

#Applying Processing to Train and Test Data
train_data = process_age(train_data, age_cut_points, age_label_names) 
train_data = process_titles(train_data, name_col, name_titles)

test_data = process_age(test_data, age_cut_points, age_label_names)
test_data = process_titles(test_data, name_col, name_titles) 

train_data.describe()

# Engineering a Feature for Family Size
# Hypothesis: The more family members, the less likely to survive

train_data_1 = train_data
train_data1["Family_size"] = train_data_1['SibSp'] + train_data_1['Parch']
train_data1.groupby(['Sex'])['Survived'].mean()
train_data1.groupby(['Sex'])['Survived'].count()

#Feature 2: Fare (Did not help)
    #Comment: Oppurtunity for model improvement in changing the location of the cuts?
    #def process_fare(df, cut_points, label_names):
    #   df["Fare"]=df["Fare"].fillna(-0.5)
    #   df['Fare_categories']= pd.cut(df['Fare'],cut_points, labels = label_names)
    #  return df
    #fare_cut_points = [-1,0,50,100,1000]
    #fare_label_names = ['Missing', 'Low', 'Medium', 'High']

    #train_data.head() = process_fare(train_data, fare_cut_points, fare_label_names) 
    #test_data = process_fare(test_data, fare_cut_points, fare_label_names) 

ac_pivot = train_data.pivot_table(index="Age_categories", values = "Survived")
ac_pivot.plot.bar()
#-------------------------------------------------------------------------------------------------------------------
#Prepping Data for Model
#-------------------------------------------------------------------------------------------------------------------
def create_dummies(df, cols):
    df = pd.get_dummies(df, columns = cols)
    return df

colms = ['Embarked', 'Pclass','Sex','Age_categories']

train_data = create_dummies(df = train_data, cols = colms)
test_data = create_dummies(df = test_data, cols = colms)

#-------------------------------------------------------------------------------------------------------------------
#Splitting the data to test the Model
#-------------------------------------------------------------------------------------------------------------------
cols_to_train = ['Sex_male',
                'Sex_female',
                'Pclass_1',
                'Pclass_2',
                'Pclass_3',
                'Name_titles',
                'Age_categories_Missing','Age_categories_Infant',
                'Age_categories_Child', 'Age_categories_Teenager',
                'Age_categories_YoungAdult', 'Age_categories_Adult',
                'Age_categories_Senior']

X = train_data.drop("Survived", axis=1)
X = X.loc[:, cols_to_train]
y= train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    stratify=y)
#-------------------------------------------------------------------------------------------------------------------
#Creating Model
#-------------------------------------------------------------------------------------------------------------------
lr = LogisticRegression()
my_model = lr.fit(X_train, y_train)
#-------------------------------------------------------------------------------------------------------------------
#Testing Model
#-------------------------------------------------------------------------------------------------------------------
#Without k-fold cross-validation
predictions = lr.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
accuracy

#With k-fold cross validation (no splitting necessary)
lr_k = LogisticRegression()
scores = cross_val_score(lr_k, X, y, cv=10) #cv is the number of folds for cross validation
np.mean(scores)
#-------------------------------------------------------------------------------------------------------------------
#Applying model to holdout test data
#-------------------------------------------------------------------------------------------------------------------
lr = LogisticRegression()
final_model = lr.fit(X, y)
holdout_prediction = lr.predict(test_data.loc[:, cols_to_train])
holdout_prediction
#-------------------------------------------------------------------------------------------------------------------