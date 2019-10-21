import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dc import titanic_data_clean
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE,RFECV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 


#returns optimal features. In the case of RFECV, returns optimal number of features as well
def feature_elimination(model,x,y,n_features_to_select = 0,type ="RFE") : 
    if(type=="RFE"):

        rfe = RFE(model,n_features_to_select)
        rfe = rfe.fit(x,y)
        return n_features_to_select,list(x.columns[rfe.support_])
    
    elif(type=="RFECV"):
        
        rfecv = RFECV(estimator = model,step = 1,cv = 10,scoring = 'accuracy')
        rfecv.fit(x,y)
        return rfecv.n_features_,list(x.columns[rfecv.support_])
    else:
        raise TypeError
        
def dump_predicted_scores(pid_test,y_pred):
    df = pd.DataFrame({'PassengerId':pid_test,'Survived':y_pred})
    df.to_csv('Predicted.csv',index = False)


df_train = pd.read_csv("titanic/train.csv")
df_test = pd.read_csv("titanic/test.csv")
pid_test = df_test["PassengerId"]
df_train = titanic_data_clean(df_train)
df_test = titanic_data_clean(df_test)


cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Embarked_Q","Pclass_3","IsFemale","IsMinor"]
x = df_train[cols]
y = df_train["Survived"]

model = LogisticRegression(solver = 'lbfgs',max_iter = 1000)

#using RFECV
n,final_features = feature_elimination(model,x,y,type = "RFECV")
x = df_train[final_features]

#heatmap
#plt.subplots(figsize=(8, 5))
#sns.heatmap(x.corr(), annot=True, cmap="RdYlGn")
#plt.show()

#remove one weak attribute
x_train,y_train = x[final_features],y
x_test = df_test[final_features]
model.fit(x_train,y_train)
y_test = model.predict(x_test)

dump_predicted_scores(pid_test,y_test)

# To understand the effect of the attributes before and after
np.random.seed(0)
X, y = x[final_features],y
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",np.logspace(-7, 3, 3),cv=5)    #Using the final features of x

train_scores_uc, valid_scores_uc = validation_curve(Ridge(), x, y, "alpha",np.logspace(-7, 3, 3),cv=5)  #Using the entirety of x







