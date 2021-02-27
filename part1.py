# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 02:21:38 2019

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt

'''To load the contents of the file into a numpy array, Here we load
both the training set and test set into data and data_test respectively '''

'''Feature train and feature test contains the data which is given into the KNN model for training purpose'''

'''train_label and test_label contains the data for prediction'''


data=np.genfromtxt(r'C:\Users\DELL\Desktop\AI\MACHINE LEARNING\Assignment1\data\classification\trainingData.csv',delimiter=',')
feature_train=data[:,:-1]
train_label=data[:,-1]

data_test=np.genfromtxt(r'C:\Users\DELL\Desktop\AI\MACHINE LEARNING\Assignment1\data\classification\testData.csv',delimiter=',')
feature_test=data_test[:,:-1]
test_label=data_test[:,-1]

'''Calculate_distance function takes 3 input paramters feature_train and feature_test and a 
parameter p which corresponds to the distance metric used (Here we will pass p=2) to compute the Eucledian 
distance, and returns the eucledian distance between the test query and the train data set'''

def Calculate_Distance(feature_train,feature_test,p):
    distance = (np.sum((feature_train - feature_test)**p,axis=1))**(1/p)
    return distance


'''Standard KNN predicts the class of the test data and the accuracy is calculated by comparing 
the predicted value aganist the orginal value'''


def Standard_KNN(feature_train,feature_test,test_label,k,p):
       counts=0
       predict=[]
       for instance in feature_test:
            distance=Calculate_Distance(feature_train,instance,p)
            #np.argsort sorts the distance and returns the indices
            indices=np.argsort(distance)
            prediction=train_label[indices]
            neighbors=prediction[0:k]
            #neighbors contains the k neighbors of the query and np.unique with count as true returns the 
            #maximum occured class into the variable value and the count into the variable count
            values,count=np.unique(neighbors,return_counts=True)
            #the majority class in the specified neighborhood k is voted as the class of the query
            predict+=[values[np.argmax(count)]]
       predicts=np.array(predict)
    
       '''Returns the no of elementwise similarity between the predicted and the orginal numpy arrays'''
       counts=np.count_nonzero(predicts==test_label)
       return (counts/1000)*100


accuracy_standard_knn=[]
x=[]
for k in range(1,20):
    print('Accuracy for k={}  for STANDARD KNN is {}'.format(k,Standard_KNN(feature_train,feature_test,test_label,k,2)))
    x+=[k]
    accuracy_standard_knn.append(Standard_KNN(feature_train,feature_test,test_label,k,2)) 
plt.title('Accuracy of STANDARD KNN for different k value')
plt.xlabel('k values')
plt.ylabel('Accuracy corresponding to k value in STANDARD KNN')
plt.plot(x,accuracy_standard_knn)
plt.show()