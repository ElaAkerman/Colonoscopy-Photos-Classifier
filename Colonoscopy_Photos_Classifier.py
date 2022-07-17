import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
df=pd.read_pickle('DigitalColposcopy.pkl')

#Organizing Data
title, x_training, x_test, y_training, y_test = df
title = title[:-1]
x_training = pd.DataFrame(x_training, columns=title)  #Transforming the train data and test data to a pd dataframe
x_test = pd.DataFrame(x_test, columns=title)

#Normalization of Data
s = StandardScaler().fit(x_training)
x_training_f = s.transform(x_training) 
x_test_f = s.transform(x_test)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def LogisticReg(X_train, Y_train, X_test, Y_test, C, threshold=0.5, steps=20, learning_rate=0.0007):
    X_train1 = X_train.shape[1]
    weight = np.zeros(X_train1)
    bias = 0 
    
    #Training the model  
    for i in range(steps): 
        Z = np.dot(weight, X_train.T) + bias
        prediction = sigmoid(Z)
    
     #Cost func. divertive 
        d_weight = np.dot( prediction-Y_train, X_train) + (2/C)*weight
        d_bias = np.sum(prediction-Y_train)
        
     #Gradient descent
        weight = weight - learning_rate*d_weight
        bias = bias - learning_rate*d_bias
        
    #Testing the model:
        Z_test = np.dot(weight, X_test.T) + bias
        prediction_test = sigmoid(Z_test)
        
        prediction_f = prediction_test>threshold # if probbility is above 0.5 the prediction is true= 1 
        prediction_f = np.array(prediction_f,dtype='float')
        
    #Accuracy and AUC:
        accuracy = accuracy_score(Y_test, prediction_f)
        AUC = roc_auc_score(Y_test, prediction_test) 
        print("After step",i+1,"the accuracy score is:",round(accuracy*100,3),"%")
        print("And the AUC is:",round(AUC,3))    
        
    return weight, bias, accuracy, AUC
  
  
