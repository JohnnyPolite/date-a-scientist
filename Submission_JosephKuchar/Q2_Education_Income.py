# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv("profiles.csv")
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


df["essay_len"] = all_essays.apply(lambda x: len(x))
df=df.dropna(axis=0, subset=['education','age','income',"essay_len"])
#print df.job.value_counts()

def sex_code(x):
    if x=='m':
        return 1
    else:
        return 2
df["sex_code"]=df.sex.map(sex_code) 
def education_level(ed):
    if ed =='graduated from college/university':
        return 2.
    elif ed=='graduated from masters program':
        return 3.        
    elif ed== 'working on college/university':
        return 1.5
         
    elif ed== 'working on masters program':
        return 2.            
    elif ed== 'graduated from two-year college':
        return 2.
    elif ed== 'graduated from high school':
        return  1.          
    elif ed== 'graduated from ph.d program':
        return  4.        
    elif ed== 'graduated from law school':
        return  2.5           
    elif ed== 'working on two-year college':
        return  1.5         
    elif ed== 'dropped out of college/university':
        return  1.5    
    elif ed== 'working on ph.d program':
        return  3.              
    elif ed== 'college/university':
        return  2.                   
    elif ed== 'graduated from space camp':
        return  1.            
    elif ed== 'dropped out of space camp':
        return   1.           
    elif ed== 'graduated from med school':
        return  2.5            
    elif ed== 'working on space camp':
        return  1.                
    elif ed== 'working on law school':
        return 1.5                 
    elif ed== 'two-year college':
        return  2.                     
    elif ed== 'working on med school':
        return 1.5
    elif ed== 'dropped out of two-year college':
        return 1.5
    elif ed== 'dropped out of masters program':
        return 2.
    elif ed== 'masters program':
        return 3.
    elif ed== 'dropped out of ph.d program':
        return 3.
    elif ed== 'dropped out of high school':
        return 0.
    elif ed== 'high school':
        return 1.
    elif ed== 'working on high school':
        return 1.
    elif ed== 'space camp':
        return 1.
    elif ed== 'ph.d program':
        return 4.
    elif ed== 'law school':
        return 2.5
    elif ed== 'dropped out of law school':
        return 1.5
    elif ed== 'dropped out of med school':
        return 1.5
    elif ed== 'med school':
        return 1.5
    elif ed== 'NaN':
        return  1.

df["education_level"]=df.education.map(education_level)  
#most reported incomes are -1 (essentially non-responding), so need to remove those
df=df.loc[df['income'] > 1.]
df=df.loc[df['income'] < 200000.]
df_ed1=df.loc[df['education_level'] == 1.]
df_ed2=df.loc[df['education_level'] == 2.]
df_ed3=df.loc[df['education_level'] >= 2.5]

plt.figure()
plt.hist(df_ed1.income,bins=50,normed=True,alpha=0.5)
plt.hist(df_ed2.income,bins=50,normed=True,alpha=0.5)
plt.hist(df_ed3.income,bins=50,normed=True,alpha=0.5)
plt.legend(['High School', 'Undergraduate', 'Post Graduate'])

#Define our X and Y data
min_max_scaler = preprocessing.MinMaxScaler()
X_temp=df[["age","education_level", "essay_len", "sex_code"]].values
X_scaled=min_max_scaler.fit_transform(X_temp)
Y=np.array(df.income)
X=pd.DataFrame(X_scaled, columns=["age","education level", "essay_len","sex_code"])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)

model1=LinearRegression()
model1.fit(X_train,Y_train)
Y_predict=model1.predict(X_test)
score=model1.score(X_test,Y_test)
print 'linear score', score
model2=KNeighborsRegressor(n_neighbors=88)
model2.fit(X_train,Y_train)
Y_predict2=model2.predict(X_test)
plt.figure(2)
plt.scatter(Y_test,Y_predict,alpha=0.5)
plt.scatter(Y_test,Y_predict2,alpha=0.5)
plt.legend(['Multilinear','K Nearest Neighbor'])
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
#plt.scatter(X_test["education level"],Y_predict)
n_vals=[]
Scores=[]
for n in range(1,200):
    model2=KNeighborsRegressor(n_neighbors=n)
    model2.fit(X_train,Y_train)
    n_vals.append(n)
    Scores.append(model2.score(X_test,Y_test))
#    print n, model2.score(X_test,Y_test)

plt.figure()
plt.scatter(n_vals,Scores)
plt.xlabel('k')
plt.ylabel('Score')

plt.show()       