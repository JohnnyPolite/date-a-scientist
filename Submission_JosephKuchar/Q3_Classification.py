import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import time
#Can we predict sex with jobs and pets?

df = pd.read_csv("profiles.csv")
df=df.dropna(axis=0, subset=['sex','pets','job'])
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
#print df.job.value_counts()

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

def pet_code(x):
    if x=='likes dogs and likes cats':
        return  5.
    elif x=='likes dogs':
        return    3.                      
    elif x=='likes dogs and has cats':
        return    7.         
    elif x=='has dogs':
        return    6.                        
    elif x=='has dogs and likes cats':
        return    8.         
    elif x=='likes dogs and dislikes cats':
        return    1.    
    elif x=='has dogs and has cats':
        return    10.           
    elif x=='has cats':
        return   4.                         
    elif x=='likes cats':
        return   2.                       
    elif x=='has dogs and dislikes cats':
        return   4.        
    elif x=='dislikes dogs and likes cats':
        return   -1.    
    elif x=='dislikes dogs and dislikes cats':
        return   -5.   
    elif x=='dislikes cats':
        return   -2.                     
    elif x=='dislikes dogs and has cats':
        return   1.        
    elif x=='dislikes dogs':
        return   -3.    

df["pet_code"]=df.pets.map(pet_code)           

def job_code(x):
    if x=='other':
        return -1
    elif x=='student':
        return  8.                            
    elif x=='science / tech / engineering':
        return  16       
    elif x=='computer / hardware / software':
        return   15    
    elif x=='artistic / musical / writer':
        return   1.       
    elif x=='sales / marketing / biz dev':
        return  2.        
    elif x=='medicine / health':
        return  14                  
    elif x=='education / academia':
        return   9.              
    elif x=='executive / management':
        return   13            
    elif x=='banking / financial / real estate':
        return   10. 
    elif x=='entertainment / media':
        return  6.              
    elif x=='law / legal services':
        return  11               
    elif x=='hospitality / travel':
        return  3.               
    elif x=='construction / craftsmanship':
        return  4.       
    elif x=='clerical / administrative':
        return  5.           
    elif x=='political / government':
        return  12              
    elif x=='rather not say':
        return    -1                    
    elif x=='transportation':
        return  6.                      
    elif x=='unemployed':
        return  -1                          
    elif x=='retired':
        return -1                              
    elif x=='military':
        return  7.                            
       
df["job_code"]=df.job.map(job_code)           

df=df.loc[df['job_code'] > 0.] #remove the values we set to -1

X= df[['job_code','pet_code']]
Y=df.sex_code
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)

t0=time.time()
classifier=KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train,Y_train)
Y_predict=classifier.predict(X_test)
t1=time.time()
print 'near neighbor time', t1-t0
print 'accuracy ', accuracy_score(Y_test,Y_predict)
print 'precision ', precision_score(Y_test,Y_predict)
print 'recall ', recall_score(Y_test,Y_predict)
print 'F1 score ', f1_score(Y_test,Y_predict)
#print np.array(X_test.pet_code)
plt.figure()
X_pets=np.array(X_test.pet_code)
X_job=np.array(X_test.job_code)
Y_pred=np.array(Y_predict)
Y_T=np.array(Y_test)
#for i in range(len(Y_pred)):
#    if Y_T[i]==1:
#        m='+'
#    else:
#        m='x'
#    if Y_pred[i]==1:
#        c='blue'
#    else:
#        c='red'
#    plt.scatter(X_pets[i],X_job[i],marker=m,color=c, alpha=0.5)
#plt.xlabel('Pet Code')
#plt.ylabel('Job Code')
#plt.show()

#Let's re-do the above with support vectors!!
#t0=time.time()
#for g in range(1,50,5):
#    
#    classifier2=SVC(gamma=g)
#    classifier2.fit(X_train,Y_train)
#    Y_predict_SVC=classifier2.predict(X_test)
#    print g, accuracy_score(Y_test,Y_predict_SVC)
#t1=time.time()
#print 'SVC time', t1-t0
print 'accuracy ', accuracy_score(Y_test,Y_predict_SVC)
print 'precision ', precision_score(Y_test,Y_predict_SVC)
print 'recall ', recall_score(Y_test,Y_predict_SVC)
print 'F1 score ', f1_score(Y_test,Y_predict_SVC)
Y_pred=np.array(Y_predict_SVC)
#for i in range(len(Y_pred)):
#    if Y_T[i]==1:
#        m='+'
#    else:
#        m='x'
#    if Y_pred[i]==1:
#        c='blue'
#    else:
#        c='red'
#    plt.scatter(X_pets[i],X_job[i],marker=m,color=c, alpha=0.5)
#plt.xlabel('Pet Code')
#plt.ylabel('Job Code')
#plt.show()