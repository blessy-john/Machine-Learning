# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 04:51:17 2019

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt


predict=[]
counts=0
data=np.genfromtxt(r'C:\Users\DELL\Desktop\AI\MACHINE LEARNING\Assignment1\data\regression\trainingData.csv',delimiter=',')
feature_train=data[:,:-1]
train_label=data[:,-1]

data_test=np.genfromtxt(r'C:\Users\DELL\Desktop\AI\MACHINE LEARNING\Assignment1\data\regression\testData.csv',delimiter=',')
feature_test=data_test[:,:-1]
test_label=data_test[:,-1]


'''Calculate_distance function takes 3 input paramters feature_train and feature_test and a 
parameter p which corresponds to the distance metric used (Here we will pass p=2) to compute the Eucledian 
distance, and returns the eucledian distance between the test query and the train data set'''

def Calculate_Distance(feature_train,feature_test,p):
    distance = (np.sum((feature_train - feature_test)**p,axis=1))**(1/p)
    return distance

      
def Distance_Weighted_KNN_Regression(feature_train,feature_test,test_label,k,p):
    
       prediction=[]  
       for instance in feature_test:
            middle_stage=[]
            d={}
            numerator=[]
            denominator=[]
            distance=Calculate_Distance(feature_train,instance,p)
            #np.argsort sorts the distance from smallest to the largest and returns the indices
            indices=np.argsort(distance)
            #target contains the classes corresponding to the indices in the indices array
            target=train_label[indices]
            #target distance contains the inverse distance wait between test and the train instance
            target_distance=distance[indices]
            #middle_stage contains all the data multiplied by the weights which is the inverse distance
            middle_stage=(target*(1/target_distance))
            #f(xq)=sum(weights*targetvalue)/sum(weights)
            #middle_stage upto k neighbors act as the numerator
            numerator=middle_stage[0:k]
            #Sum of the distance act as the denominator
            denominator=1/target_distance[0:k]
           
            
            prediction+=[sum(numerator)/sum(denominator)]
       predicts=np.array(prediction)
       
       #computation of r square          
       return 1-((np.sum(np.square(test_label-predicts)))/np.sum(np.square((test_label-np.mean(test_label)))))
   
k_desired=int(input('Enter the desired k value'))
print(Distance_Weighted_KNN_Regression(feature_train,feature_test,test_label,k_desired,2))

accuracy_distance_weighted_knn=[]
k_values=[] 

#calculating accuracy of distance weighted knn for different value of k 
   
#for k in range(1,8):
#    k_values+=[k]
#    print('accuarcy of Distance Weighted KNN Regressor with k {} is {}'.format(k,Distance_Weighted_KNN_Regression(feature_train,feature_test,test_label,k,2)))
#    accuracy_distance_weighted_knn+=[Distance_Weighted_KNN_Regression(feature_train,feature_test,test_label,k,2)]

#plt.title('Accuracy for distance weighted knn regressor')
#plt.xlabel('k_values')
#plt.ylabel('Accuracy for different k values')
#plt.plot(k_values,accuracy_distance_weighted_knn)
#plt.show()

#-----------------------------------------------PART(ii)----------------------------------------------------------------------

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


knn=KNeighborsRegressor()
knn.fit(feature_train,train_label)
predict=knn.predict(feature_test)
'''Benchmark is the r2 score without removing any of the features and act as the benchmark'''
benchmark=r2_score(predict,test_label)
print('The benchmark for Kneighbor Classifier is {}'.format(benchmark))

for m in range(0,11):
    data_check=feature_train
    '''The data is fitted by deleting each of the feature in range from 1st feature
    to the eleventh feature and the r2 score is computed'''
    knn.fit(np.delete(data_check,m,axis=1),train_label)
    predicted_value=knn.predict(np.delete(feature_test,m,1))

    score=r2_score(predicted_value,test_label)
    print('R2 score by excludihg the {} feature from training set is {}'.format(m,score))

#____________________ Accuracy by removing Irrelevant feautures______________________________
'''We now have come to know about the errelevant features so lets remove the errelevant
features and compute the r2 score'''
x=np.delete(feature_train,np.s_[0:6],1)
x_test=np.delete(feature_test,np.s_[0:6],1)
knn.fit(x,train_label)
selected_prediction=knn.predict(x_test)
score=r2_score(selected_prediction,test_label)
print('R2 score by removing the irrelevant features in distance weighted regressor is {}'.format(score))

#_________________________________Cross validation_______________________________

'''For cross validation the entire data is splitted so lets combine the test and training data'''
X=np.vstack((feature_train,feature_test))
Y=np.concatenate((train_label,test_label),axis=0)

'''For each fold the r2 score is computed'''
K_FOLD=[]
final=[]
for k_fold in range(2,20):
    K_FOLD+=[k_fold]
    score_cross_val=np.mean(cross_val_score(knn, X , Y,cv=k_fold, scoring='r2'))
    final+=[score_cross_val]
print(np.mean(final))
plt.title('Cross_validation')
plt.xlabel('k splits')
plt.ylabel('r2 score')
plt.plot(K_FOLD,final)
plt.show()
#____________________________________Grid Search CV__________________________________________
'''To find the best value of knn and to get the best score we can simply use grid search'''

param_grid={}
from sklearn.model_selection import GridSearchCV
k_range=list(range(1,31))
param_grid=dict(n_neighbors=k_range)

grid=GridSearchCV(knn,param_grid,cv=10,scoring='r2')
grid.fit(X,Y)
print('The Best score of the model is {}'.format(grid.best_score_))
print('The best parameters for the model is {}'.format(grid.best_params_))
print(grid._get_param_names)
