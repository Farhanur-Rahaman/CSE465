#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score


# In[2]:


df = pd.read_csv("E:\DRISHTY NSU\cse465\indian_liver_patient.csv")


# In[3]:


df.info()


# In[4]:


df.describe(include='all')


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


df[df['Albumin_and_Globulin_Ratio'].isnull()]


# In[7]:


df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())


# In[8]:


df.isnull().sum()


# In[9]:


import warnings
warnings.filterwarnings('ignore')


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[15]:


from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits=5,random_state=42)
logmodel = LogisticRegression(C=1, penalty='l1')
results = cross_val_score(logmodel, X_train,y_train,cv = kfold)
print(results)
print("Accuracy:",results.mean()*100)


# In[16]:


import warnings
warnings.filterwarnings('ignore')


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[20]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
print(confusion_matrix(y_test,rf_predicted))
print(classification_report(y_test,rf_predicted))


# In[21]:


import warnings
warnings.filterwarnings('ignore')


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[25]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
#Predict Output
gauss_predicted = gaussian.predict(X_test)

gauss_score = round(gaussian.score(X_train, y_train) * 100, 2)
gauss_score_test = round(gaussian.score(X_test, y_test) * 100, 2)
print('Gaussian Score: \n', gauss_score)
print('Gaussian Test Score: \n', gauss_score_test)
print('Accuracy: \n', accuracy_score(y_test, gauss_predicted))
print(confusion_matrix(y_test,gauss_predicted))
print(classification_report(y_test,gauss_predicted))

sns.heatmap(confusion_matrix(y_test,gauss_predicted),annot=True,fmt="d")


# In[70]:


plt.figure(figsize=(8,3))
p = sns.lineplot(range(1,15),gauss_score,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),gauss_score_test,marker='o',label='Test Score')


# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[29]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[32]:


library(readr)
df = pd.read_csv("E:\DRISHTY NSU\cse465\indian_liver_patient.csv")

#converting neumeric variable to categorical variable
df["class"] = ifelse(df$Dataset == 2,"not_patient","patient")
# 1 is patient
# 2 is not patient

#creating dummy variable for Gender
df["female"] <- ifelse(df$Gender == "Female",1,0)
df["Male"] <- ifelse(df$Gender == "Male",1,0)

#checking missing values
sum(is.na(df))

#checking which variable contains missing values
summary(df)

#detecting outliers
boxplot(df$Albumin_and_Globulin_Ratio)
#Albumin_and_Globulin_Ratio has 4 missing values

#Replacing with median because Albumin_and_Globulin_Ratio contains outliers
ag_median <- median(df$Albumin_and_Globulin_Ratio,na.rm = T)

df$Albumin_and_Globulin_Ratio[is.na(df$Albumin_and_Globulin_Ratio)] <- ag_median
sum(is.na(df))

#Creating levels for Class variable
df["Class"] <- factor(df$Class)

#normalize the variables
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
df_norm<-as.data.frame(lapply(df[,-c(2,11:14)],FUN=normalize))
liver <- data.frame(df_norm,df[,c(13,14,12)]) 

liver_patient <- liver[liver$Class == "patient",]
liver_not_patient <- liver[liver$Class == "not_patient",]

#train-test spleet
liver_train <- rbind(liver_patient[1:291,],liver_not_patient[1:117,])
liver_test <- rbind(liver_patient[292:416,],liver_not_patient[118:167,])

# Using multilayered feed forward nueral network
# package nueralnet
# install.packages("neuralnet")
library(neuralnet)

# Building model
formula_nn <- Class~Age+Total_Bilirubin+Direct_Bilirubin+Alkaline_Phosphotase+Alamine_Aminotransferase+Aspartate_Aminotransferase+Total_Protiens+Albumin+Albumin_and_Globulin_Ratio+female+Male
liver_model <- neuralnet(formula_nn,data=liver_train)
str(liver_model)
plot(liver_model)


set.seed(7)
model_results <- predict(liver_model,liver_test[1:11]) #we use predict instead of compute
library(dplyr)
predicted_size_category <- ifelse(model_results[,1] > model_results[,2],"not_patient","patient")
# model_results$neurons
mean(predicted_size_category==liver_test$Class) #Testing Accuracy = 73% 


# In[33]:


import warnings
warnings.filterwarnings('ignore')


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[36]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[52]:


import warnings
warnings.filterwarnings('ignore')


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[55]:


def KNN(X_train, y_train, X_test, y_test):
    
    reg=KNeighborsClassifier(n_neighbors=8)

    #lr = GridSearchCV(clf, param_grid=grid_values, scoring="f1")

    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)

    grid = GridSearchCV(reg, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid.fit(X_train, y_train)
       
    print('Accuracy of KNeighbors classifier on test set: {:.2f}'.format(grid.score(X_test, y_test)))
    print ("Classification report:\n{}".format(classification_report(y_test,grid.predict(X_test))))
    
    precision,recall,fscore,support=score(y_test, grid.predict(X_test))
    return fscore, accuracy_score(y_test, grid.predict(X_test))


# In[56]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)



fscore, accuracy = KNN(X_train, y_train, X_test, y_test)

fsc1.append(fscore[0])
fsc2.append(fscore[1])
acc.append(accuracy)



# In[57]:


import warnings
warnings.filterwarnings('ignore')


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[60]:


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[61]:


max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[62]:


max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[63]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[86]:


import warnings
warnings.filterwarnings('ignore')


# In[87]:


X=df[['Age','Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Total_Protiens', 'Albumin']]
X=df.drop("Dataset",axis=1)
y=df["Dataset"]


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=0.3, random_state=123)


# In[90]:


from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_blobs


# In[91]:


model=LogisticRegression(multi_class='multinomial',solver='lbfgs')


# In[92]:


model.fit(X_train, y_train)


# In[93]:


import warnings
warnings.filterwarnings('ignore')


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[97]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[98]:


from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_blobs


# In[99]:


model=LogisticRegression(multi_class='multinomial',solver='lbfgs')


# In[100]:


model.fit(X_train, y_train)


# In[101]:


pred=model.predict(X_test)


# In[102]:


confusion_matrix(y_test,pred)


# In[103]:


from sklearn.metrics import accuracy_score


# In[104]:


accuracy_score(y_test,pred)


# In[105]:


pred=model.predict(X_train)
accuracy_score(y_train,pred)


# In[ ]:




