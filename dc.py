import numpy as np
import pandas as pd

#data cleaning
def titanic_data_clean(dataset) : 
    
    dataset.replace({'male':0,'female':1},inplace = True)
    dataset.rename({"Sex":"IsFemale"},axis = 1,inplace = True)
    
    #drop useless columns
    dataset.drop(["PassengerId","Name","Ticket","Cabin"],axis = 1,inplace = True)
            
    #Fill NaN age values with median
    dataset["Age"].fillna(dataset["Age"].median(skipna = True),inplace = True)
    
    #fill missing Embarked column values with the most common occurence
    dataset["Embarked"].fillna(dataset['Embarked'].value_counts().idxmax(), inplace=True)
    
    #combine Sibsp and Parch to one variable
    dataset['TravelAlone']=np.where((dataset["SibSp"]+dataset["Parch"])>0, 0, 1)
    dataset.drop('SibSp', axis=1, inplace=True)
    dataset.drop('Parch', axis=1, inplace=True)
    
    #create categorical variables and drop some variables
    dataset=pd.get_dummies(dataset, columns=["Pclass","Embarked"])
    
    dataset['IsMinor']=np.where(dataset['Age']<=16, 1, 0)
    
    return dataset


