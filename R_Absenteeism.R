#Load all the required libraries that are needed to compute
library(dplyr)
library(corrplot)
library(DMwR)
library(caTools)
library(corrgram)
library(e1071)
library(rpart)
library(randomForest)
library(rsq)

#After converting the excel to csv
#Read the data to the dataframe to be used in R Studio
df=read.csv('R_Absenteeism_at_work_Project.csv', header = TRUE)

#Check the summary of the data
summary(df)
#Check the structue of the data
str(df)
#Check the header of the data to get insight of the data
head(df, n=5)

#Add all the column names to the list and make
#two separate vectors for numerical and categorical list
columnsList = c("ID", "Reason.for.absence", 'Month.of.absence', 'Day.of.the.week', 
               'Seasons', 'Transportation.expense', 
               'Distance.from.Residence.to.Work', 'Service.time', 'Age',
               'Work.load.Average.day ', 'Hit.target', 'Disciplinary.failure',
               'Education', 'Son', 'Social.drinker', 'Social.smoker', 'Pet',
               'Weight', 'Height', 'Body.mass.index', 'Absenteeism.time.in.hours')
numericalColumnsList = c('Transportation.expense', 'Distance.from.Residence.to.Work',
                        'Service.time', 'Age', 'Work.load.Average.day', 'Son',
                        'Pet', 'Weight', 'Height', 'Body.mass.index', 'Hit.target',
                        'Absenteeism.time.in.hours')
categoricalColumnsList =c('ID', 'Reason.for.absence', 'Month.of.absence', 'Day.of.the.week', 
                         'Seasons', 'Disciplinary.failure', 'Education',
                         'Social.drinker', 'Social.smoker')

#Replace all the data of Reason and months which contains 0 
#with NA as this values does not exist
df$Reason.for.absence[which(df$Reason.for.absence==0)] = NA
df$Month.of.absence[df$Month.of.absence==0]=NA

#Convert all the columns that are categorical but has type numerical
#Converting the type to factors
for(i in categoricalColumnsList){
  print(i)
  df[i]=lapply(df[i], factor)
}

#Plot all the boxplots of the numerical data columns
for(i in numericalColumnsList){
  tempStr= paste('Boxplot for',i, sep = ' ')
  boxplot(df[i],
          main = tempStr,
          col='orange',
          border='brown',
          horizontal = TRUE,
          notch = TRUE)
}

#Impute all the data points which contains NA as their value
knnOutput=knnImputation(df, k=3, scale = FALSE)
df_clean = knnOutput
anyNA(df)
colnames(df_clean)[!complete.cases(t(df_clean))]
str(df_clean)

#Plot the correlation plot of the numerical data to check
#their correlation
corrplot(cor(df_clean[sapply(df_clean, is.numeric)]))

#Remove all the variables that are highly correlated 
df_clean$Weight=NULL
df_clean$Age=NULL
df_clean$Service.time=NULL

#As per further obsercations it was found that Pet and Social Smoker column
#are correlated to other features
df_clean$Pet=NULL
df_clean$Social.smoker=NULL

#Removing all the names in the vector numericalColumnsList that are removed from 
#the dataframe
numericalColumnsList=numericalColumnsList[!numericalColumnsList == 'Pet']
numericalColumnsList=numericalColumnsList[!numericalColumnsList == 'Weight']
numericalColumnsList=numericalColumnsList[!numericalColumnsList == 'Age']
numericalColumnsList=numericalColumnsList[!numericalColumnsList == 'Service.time']

#Splitting all the training and test
split = sample.split(df$Absenteeism.time.in.hours, SplitRatio = 0.8)
training_set  = subset(df_clean, split = TRUE)
test_set = subset(df_clean, split = FALSE)

#Scaling the numerical data in training and test set
for( i in numericalColumnsList){
  print(i)
  training_set[,i]= scale(training_set[,i])
  test_set[,i]= scale(test_set[,i])
}

#Linear Regression
linearRegressor = lm(formula = Absenteeism.time.in.hours ~.-Education,
                     data = training_set)
linPred = predict(linearRegressor, newdata = test_set)
summary(linearRegressor)

#R square value of the linear Regression
rsq(linearRegressor)

#Support Vector Regression
svReg = svm(formula= Absenteeism.time.in.hours~.,
            data = df_clean,
            type = 'eps-regression')
svPred = predict(svReg, df_clean)
summary(svReg)

#Decision Tree Regression
dtRegressor = rpart(formula = Absenteeism.time.in.hours~.,
                    data = training_set)
dtPred = predict(dtRegressor, test_set)
summary(dtPred)

#Now we look at the loss for the year 2011
#We Are assuming that the data is collected for 1 year 
totalABShours=sum(df_clean$Absenteeism.time.in.hours)
totalWorkload=sum(df_clean$Work.load.Average.day)
hitMean=mean(df_clean$Hit.target)
totalWorkloadWasted=totalWorkload-((hitMean*totalWorkload)/100)

print('Projection of loss in 2011:')
print(paste('Total Absenteeism in hours: ',totalABShours))
print(paste('Total worklaod missed: ', totalWorkloadWasted))

