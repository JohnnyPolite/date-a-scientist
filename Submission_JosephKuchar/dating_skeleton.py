import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv("profiles.csv")
df=df.dropna(axis=0, subset=['height','education'])
#print df.education.value_counts()
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


#Question: Do short men need to put in more effort on dating sites?
#If that's the case, there should be a (negative) correlation with amount of detail given in essay answers with height among males.

#Create height categories for very short, short, 'normal', tall, and very tall
#<5, 5 - 5'8, ,5'8-6.,6'1-6'4,>6'4
def Height_Rank(h):
    if h<=60:
        return 0
    elif h<=68:
        return 1
    elif h<=72:
        return 2
    elif h<=76:
        return 3
    else:
        return 4
df["height_code"]=df.height.map(Height_Rank)

#print df.height.head()
#print df.height_code.head()  

# We have created (somewhat arbitrary) height rankings
#Now we can use essay text length as a measure of 'effort'  

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]


# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


df["essay_len"] = all_essays.apply(lambda x: len(x))

#print df.sex.head()

#I'm interested in males specifically, so let's do this:

df_men=df.loc[df['sex'] == 'm']

plt.figure(1)
plt.scatter(df_men.height,df_men.essay_len)
plt.xlabel("Height (inches)")
plt.ylabel("Combined Essay Length")
#plt.show()
#This does not suggest a linear relationship, but we can try all the same.

min_max_scaler = preprocessing.MinMaxScaler()
X_temp=df_men[["height","education_level"]].values
X_scaled=min_max_scaler.fit_transform(X_temp)
Y=np.array(df_men.essay_len)
X=pd.DataFrame(X_scaled, columns=["height","education level"])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42)

model1=LinearRegression()
model1.fit(X_train,Y_train)
Y_predict=model1.predict(X_test)
score=model1.score(X_test,Y_test)
print 'linear score', score

#now let's try a nearest neighbour regressor
#for n in range(1,100):
#    model2=KNeighborsRegressor(n_neighbors=n)
#    model2.fit(X_train,Y_train)
#    print n, model2.score(X_test,Y_test)
model2=KNeighborsRegressor(n_neighbors=37)
model2.fit(X_train,Y_train)
predict_2=model2.predict(X_test)
plt.figure(2)
plt.scatter(X_test["height"],Y_test,alpha=0.5)
plt.scatter(X_test["height"],Y_predict)
plt.scatter(X_test["height"],predict_2)
plt.legend(['Test Samples', 'Linear Predictions', 'NN Predictions'])
plt.xlabel("Height (normalised)")
plt.ylabel("Combined Essay Length")
#One issue here is that the line is biased towards the region with the most samples
#should create normalised frequency histogram

#make new column:verbose, 1 or 0 (more than average, less than average)

avg_word=Y.mean()

def is_verbose(x):
    if x>avg_word:
        return 1
    else:
        return 0

df_men["verbose"]=df_men.essay_len.map(is_verbose)

print pd.crosstab(df_men.height_code, df_men.verbose)

H=[0,1,2,3,4]
#numbers from crosstab
No_verbose=np.array([46.,5594.,11569.,4605.,347.])
Yes_verbose=np.array([17.,3350.,7254.,2833.,214.])
plt.figure(3)
plt.scatter(H,Yes_verbose/(Yes_verbose+No_verbose))
plt.xticks(H,['VS','S','A','T','VT'])
plt.xlabel('Height Category')
plt.ylabel('Relative Frequency of Long Essays')

#plt.figure(4)
#plt.hist(df.education_level.dropna(), bins=8)
#plt.xlabel('Education Level')
#plt.ylabel('Frequency')
#
#plt.figure(5)
#plt.hist(df.age.dropna(), bins=20)
#plt.xlabel('Age')
#plt.ylabel('Frequency')
#
#plt.figure(6)
#plt.hist(df.height.dropna(), bins=20)
#plt.xlabel('Height (inches)')
#plt.ylabel('Frequency')
#I'm pretty convinced the result here is a negative one. Let's try one more regression problem
#Q2: Can we predict income based on combined essay length + education level?
plt.show()