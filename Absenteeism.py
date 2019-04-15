#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:39:27 2019

@author: gauravmalik
"""
#Importing all the libraries that will be needed in this project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from fancyimpute import KNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#Extracting the data to which we will be working
df = pd.read_excel('Absenteeism_at_work_Project.xls')

#Getting the information about all the columns in the dataframe
df.info()

#Looking at the dataframe features information once again
df.info()

#Storing all the column names in a list if we need it just in case
#Storing all the numerical and categorical columns in seperate lists in case we need
columnsList = ['ID', 'Reason for absence', 'Month of absence', 'Day of the week', 
               'Seasons', 'Transportation expense', 
               'Distance from Residence to Work', 'Service time', 'Age',
               'Work load Average/day ', 'Hit target', 'Disciplinary failure',
               'Education', 'Son', 'Social drinker', 'Social smoker', 'Pet',
               'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']
numericalColumnsList = ['Transportation expense', 'Distance from Residence to Work',
                        'Service time', 'Age', 'Work load Average/day ', 'Son',
                        'Pet', 'Weight', 'Height', 'Body mass index', 'Hit target',
                        'Absenteeism time in hours']
categoricalColumnsList =['ID', 'Reason for absence', 'Month of absence', 'Day of the week', 
                         'Seasons', 'Disciplinary failure', 'Education',
                         'Social drinker', 'Social smoker']

#Defining funciton uniqueCount to check the unique values in categorical features
#Parameter::::>>>list of column to check in the dataframe
def uniqueCount(column):
    for i in range(len(column)):
        uniqueCount=df[column[i]].unique()
        print column[i],' : ', uniqueCount
uniqueCount(categoricalColumnsList)

#As there is no value of zero in reason of absence and also there is no
#month with numerical value zero. Just replace them with zero
df['Reason for absence']= df['Reason for absence'].replace(0, np.nan)
pd.to_numeric(df['Reason for absence'], errors= 'coerce')
df['Month of absence']=df['Month of absence'].replace(0, np.nan)
pd.to_numeric(df['Month of absence'], errors = 'coerce')

#Distingushing all the categorical features by converting their type to category
df['Reason for absence']=df['Reason for absence'].astype('category')
df['Month of absence']=df['Month of absence'].astype('category')
df['Day of the week']=df['Day of the week'].astype('category')
df['Seasons']=df['Seasons'].astype('category')
df['Disciplinary failure']=df['Disciplinary failure'].astype('category')
df['Education']=df['Education'].astype('category')
df['Social drinker']=df['Social drinker'].astype('category')
df['Social smoker']=df['Social smoker'].astype('category')

#Once again checking the unique values in categorical features
uniqueCount(categoricalColumnsList)

#Check the covariance with the plot
corr = df.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, xticklabels = corr.columns.values,
            yticklabels = corr.columns.values, annot =True,
            ax=ax)
plt.tight_layout()
plt.savefig(fname='corrplot.png')

#Dropping the weight column due to high collinearity
toDrop = ['Weight', 'Age', 'Service time']
df_drop=df.drop(columns=toDrop, axis=1)
numericalColumnsList.remove('Weight')
numericalColumnsList.remove('Age')
numericalColumnsList.remove('Service time')
columnsList.remove('Weight')
columnsList.remove('Age')
columnsList.remove('Service time')
#Checking the new dataframe df_drop
df_drop.info()
df_drop.describe()
#Looking at the sum of null values in all the features
df_drop.isnull().sum().sum

#Plotting to check all the outliers in the dataset
for i in range(len(numericalColumnsList)):
    column = numericalColumnsList[i]
    fig=plt.subplots(figsize=(5,5))
    sns.boxplot(data=df_drop[column])
    sns.swarmplot(data=df_drop[column], color='0.25')
    if(column=='Work load Average/day '):
        column= 'Workload'
    fname='Boxplot'+column+'.png'
    plt.title(column)
    plt.savefig(fname=fname)
    plt.show()
#For removing all the outliers in the numerical data 
#funtion replace_outliers accepts one column at a time and a numerical value 
#which is multiplied with standard deviation of the column values
def replace_outliers(column, std):
    column[np.abs(column- column.mean()) >std*column.std()]= np.nan
    return column

#Since we only check the outliers value for continuous variables we must seperate 
#this data from the rest
numericalOutliersColumnsList=['Transportation expense', 'Distance from Residence to Work',
                              'Work load Average/day ', 'Son', 'Pet', 'Height', 
                              'Body mass index', 'Hit target','Absenteeism time in hours']
df_outliers = df_drop[numericalOutliersColumnsList]
df_drop[numericalOutliersColumnsList] = df_outliers.transform(lambda c : replace_outliers(c, 2))
df_drop.isnull().sum().sum
#Removing all the oultiers in the target feature so that it won't hurt our prediction model
df_drop = df_drop.drop(df_drop[df_drop['Absenteeism time in hours'].isnull()].index, axis=0)

#Using KNN imputation method to impute all the null values in the data frame
df_complete = pd.DataFrame(KNN(k = 3).fit_transform(df_drop), columns = df_drop.columns)
df_complete = np.around(df_complete)#, decimals=0)
df_complete.isnull().sum().sum
df_complete.info()
df_complete.isnull().sum().sum
df_complete['Work load Average/day '].sum()


df_X = df_complete.iloc[:, :-1].values
df_y = df_complete.iloc[:, 17].values

#Now lets check the svr after splitting the train and test data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.2)


#Standard Scaler
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

#Fitting svr
svr = SVR(kernel='sigmoid')
svr.fit(X_train, y_train)

svrPred = svr.predict(X_test)

print('Score mse SVR: %.2f'%mean_squared_error(y_test, svrPred))
print ('Score of SVR(r2): %.2f'%r2_score(y_test, svrPred))

#fitting decision tree
dtReg = DecisionTreeRegressor(max_depth = 3)
dtReg.fit(X_train, y_train)

dtPred = dtReg.predict(X_test)

print('Score mse Decision Tree: %.2f'%mean_squared_error(y_test, dtPred))
print ('Score of Decision Tree(r2): %.2f'%r2_score(y_test, dtPred))

#fitting random forest
rnfReg = RandomForestRegressor(n_estimators =5, max_depth =2)
rnfReg.fit(X_train, y_train)

rnfPred = rnfReg.predict(X_test)

print('Score mse Random Forest: %.2f'%mean_squared_error(y_test, rnfPred))
print('Score of Random Forest(r2): %.2f'%r2_score(y_test, rnfPred))

#for linear multiple regression
linearReg = LinearRegression()
linearReg.fit(X_train, y_train)

linearPred=linearReg.predict(X_test)

print('Score mse Multiple Linear Regression: %.2f'%mean_squared_error(y_test, linearPred))
print('Score of Linear Regression(r2): %.2f'%r2_score(y_test, linearPred))

#Applying k fold cross validation
accuracies = cross_val_score(estimator = linearReg, X= X_train, y=y_train, cv= 10)
print('Accuracy of LinearRegression: %.2f'%accuracies.mean())

#To predict the loss in 2011
#For now it is not sure how long the data was recorded so a fucntion is made 
#In predictingLoss function the parameter is the number of years the data was observed
df_ForLossInsight = pd.DataFrame(columns=['Month', 'Total Absenteeism Hour',
                                          'Target(Workload Missed)'])
def predictingLoss(n):
    c=1
    for i in range(12):
        temp = c+i
        df_temp = df_complete[df_complete['Month of absence'] == temp]
        TotalAbsentHours=df_temp['Absenteeism time in hours'].sum()/n
        workload_total = df_temp['Work load Average/day '].sum()/n
        hit_target_mean = df_temp['Hit target'].mean()/n
        targetMissed =workload_total- (hit_target_mean*workload_total)/100
        global df_ForLossInsight
        df_ForLossInsight=df_ForLossInsight.append({'Month':temp, 'Total Absenteeism Hour':TotalAbsentHours,
                                          'Target(Workload Missed)':targetMissed}, ignore_index = True)

#If the number of years was 1 then replace 3 with 1 and get the result accordingly        
predictingLoss(3)
print('Expencted loss in 2011:')
print('Absenteeism time in hours: %.2f'%df_ForLossInsight['Total Absenteeism Hour'].sum())
print('Target Missed Total: %.2f'%df_ForLossInsight['Target(Workload Missed)'].sum())