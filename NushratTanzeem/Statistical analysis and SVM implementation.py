#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv("E:\DRISHTY NSU\cse465\indian_liver_patient.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe(include='all')


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


sns.countplot(data=df, x = 'Dataset', label='Count')

LD, NLD = df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[9]:


sns.countplot(data=df, x = 'Gender', label='Count')

M, F = df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)


# In[10]:


sns.factorplot(x="Age", y="Gender", hue="Dataset", data=df);


# In[11]:


sns.factorplot(x="Total_Bilirubin", y="Gender", hue="Dataset", data=df);


# In[12]:


sns.factorplot(x="Direct_Bilirubin", y="Gender", hue="Dataset", data=df);


# In[13]:


sns.factorplot(x="Alkaline_Phosphotase", y="Gender", hue="Dataset", data=df);


# In[14]:


sns.factorplot(x="Alamine_Aminotransferase", y="Gender", hue="Dataset", data=df);


# In[15]:


sns.factorplot(x="Aspartate_Aminotransferase", y="Gender", hue="Dataset", data=df);


# In[16]:


sns.factorplot(x="Total_Protiens", y="Gender", hue="Dataset", data=df);


# In[17]:


sns.factorplot(x="Albumin", y="Gender", hue="Dataset", data=df);


# In[18]:


sns.factorplot(x="Albumin_and_Globulin_Ratio", y="Gender", hue="Dataset", data=df);


# In[19]:


df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)


# In[20]:


df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)


# In[21]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[22]:


g = sns.FacetGrid(liver_df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Total_Bilirubin", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[23]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Total_Bilirubin", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[24]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Direct_Bilirubin", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[25]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Alkaline_Phosphotase", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[26]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Alamine_Aminotransferase", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[27]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Aspartate_Aminotransferase", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[28]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Total_Protiens", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[29]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Albumin", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[30]:


g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Albumin_and_Globulin_Ratio", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[31]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[32]:


sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")


# In[33]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[34]:


sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")


# In[35]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[36]:


sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=df, kind="reg")


# In[37]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[38]:


sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")


# In[39]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[40]:


sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=df, kind="reg")


# In[41]:


g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")
plt.subplots_adjust(top=0.9)


# In[43]:


sns.jointplot("Albumin_and_Globulin_Ratio", "Total_Protiens", data=df, kind="reg")


# In[44]:


df_rank = df.groupby(['Age'])  
df_rank.max()


# In[45]:


df_rank = df.groupby(['Age'])  
df_rank.max()


# In[46]:


df_rank = df.groupby(['Total_Bilirubin'])  
df_rank.max()


# In[47]:


df_rank = df.groupby(['Direct_Bilirubin'])  
df_rank.max()


# In[48]:


df_rank = df.groupby(['Alkaline_Phosphotase'])  
df_rank.max()


# In[49]:


df_rank = df.groupby(['Alamine_Aminotransferase'])  
df_rank.max()


# In[50]:


df_rank = df.groupby(['Aspartate_Aminotransferase'])  
df_rank.max()


# In[51]:


df_rank = df.groupby(['Total_Protiens'])  
df_rank.max()


# In[52]:


df_rank = df.groupby(['Albumin'])  
df_rank.max()


# In[53]:


df_rank = df.groupby(['Albumin_and_Globulin_Ratio'])  
df_rank.max()


# In[ ]:


plt.xlabel("Albumin_and_Globulin_Ratio")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[54]:


df.groupby('rank')[['Total_Bilirubin']].mean()


# In[55]:


plt.xlabel("Albumin_and_Globulin_Ratio")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[56]:


plt.xlabel("Total_Bilirubin")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[57]:


plt.xlabel("Direct_Bilirubin")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[58]:


plt.xlabel("Alkaline_Phosphotase")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[59]:


plt.xlabel("Alamine_Aminotransferase")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[60]:


plt.xlabel("Aspartate_Aminotransferase")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[61]:


plt.xlabel("Total_Protiens")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[62]:


plt.xlabel("Albumin")
plt.ylabel("Dataset")
plt.plot(df.Dataset,df.Albumin_and_Globulin_Ratio)
plt.show


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X=df[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[65]:


from sklearn.model_selection import train_test_split


# In[66]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[69]:


len(X_train)


# In[70]:


len(X_test)


# In[71]:


model.fit(X_train, y_train)


# In[72]:


import warnings
warnings.filterwarnings('ignore')


# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[76]:


len(X_train)


# In[77]:


len(X_test)


# In[78]:


from sklearn.svm import SVC
model = SVC()


# In[79]:


model.fit(X_train, y_train)


# In[80]:


df[df['Albumin_and_Globulin_Ratio'].isnull()]


# In[81]:


df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())


# In[83]:


df.isnull().sum()


# In[84]:


import warnings
warnings.filterwarnings('ignore')


# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[88]:


len(X_train)


# In[89]:


len(X_test)


# In[90]:


from sklearn.svm import SVC
model = SVC()


# In[91]:


model.fit(X_train, y_train)


# In[92]:


model.score(X_test, y_test)


# In[93]:


model.score(X_train, y_train)


# In[94]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[96]:


len(X_train)


# In[97]:


len(X_test)


# In[98]:


from sklearn.svm import SVC
model = SVC()


# In[99]:


model.fit(X_train, y_train)


# In[100]:


model.score(X_test, y_test)


# In[101]:


model.score(X_train, y_train)


# In[102]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[104]:


len(X_train)


# In[105]:


len(X_test)


# In[106]:


from sklearn.svm import SVC
model = SVC()


# In[107]:


model.fit(X_train, y_train)


# In[108]:


model.score(X_test, y_test)


# In[109]:


model.score(X_train, y_train)


# In[110]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin']]
y=df['Dataset']


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[112]:


len(X_train)


# In[113]:


len(X_test)


# In[114]:


from sklearn.svm import SVC
model = SVC()


# In[115]:


model.fit(X_train, y_train)


# In[116]:


model.score(X_test, y_test)


# In[117]:


model.score(X_train, y_train)


# In[118]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[119]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[120]:


len(X_train)


# In[121]:


len(X_test)


# In[122]:


from sklearn.svm import SVC
model = SVC()


# In[123]:


model.fit(X_train, y_train)


# In[124]:


model.score(X_test, y_test)


# In[125]:


model.score(X_train, y_train)


# In[126]:


X=df[['Age', 'Total_Bilirubin',
       'Alkaline_Phosphotase',
       'Aspartate_Aminotransferase', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[127]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[128]:


len(X_train)


# In[129]:


len(X_test)


# In[130]:


from sklearn.svm import SVC
model = SVC()


# In[131]:


model.fit(X_train, y_train)


# In[132]:


model.score(X_test, y_test)


# In[133]:


model.score(X_train, y_train)


# In[134]:


X=df[['Age', 'Total_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[136]:


len(X_train)


# In[137]:


len(X_test)


# In[138]:


from sklearn.svm import SVC
model = SVC()


# In[139]:


model.fit(X_train, y_train)


# In[140]:


model.score(X_test, y_test)


# In[141]:


model.score(X_train, y_train)


# In[142]:


X=df[['Age', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[144]:


len(X_train)


# In[145]:


len(X_test)


# In[146]:


from sklearn.svm import SVC
model = SVC()


# In[147]:


model.fit(X_train, y_train)


# In[148]:


model.score(X_test, y_test)


# In[149]:


model.score(X_train, y_train)


# In[150]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[151]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[152]:


len(X_train)


# In[153]:


len(X_test)


# In[154]:


from sklearn.svm import SVC
model = SVC()


# In[155]:


model.fit(X_train, y_train)


# In[156]:


model.score(X_test, y_test)


# In[157]:


model.score(X_train, y_train)


# In[158]:


X=df[['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']]
y=df['Dataset']


# In[159]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=123)


# In[160]:


len(X_train)


# In[161]:


len(X_test)


# In[162]:


from sklearn.svm import SVC
model = SVC()


# In[163]:


model.fit(X_train, y_train)


# In[164]:


model.score(X_test, y_test)


# In[165]:


model.score(X_train, y_train)


# In[ ]:




