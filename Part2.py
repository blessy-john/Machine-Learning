# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 04:18:00 2019

@author: DELL
"""

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



def Calculate_Distance(feature_train,feature_test,p):
    '''Equation coressponds to minkowski distance, p=1 ===> manhattan distance p=2 ==> eucledian distance and so on'''
    distance = (np.sum((feature_train - feature_test)**p,axis=1))**(1/p)
    return distance


''' Distance_weighted_knn takes 4 parameters that is the train data , the test data ,
 value of k and the parameter p for computing distance when p=2 its the eucledian distance,p=1 manhattan 
 distance and returns the accuracy'''

def Distance_Weighted_KNN(feature_train,feature_test,test_label,k,p):
      counts=0
      prediction=[]
      for instance in feature_test:
            middle_state=[]
            d={}
            '''Calculates the eucledian distance between each instance of the test set with the entire training set '''
            distance=Calculate_Distance(feature_train,instance,p)
            #np.argsort sorts the distance from smallest to the largest and returns the indices
            indices=np.argsort(distance)
            #target contains the classes corresponding to the indices in the indices array
            target=train_label[indices]
            #target distance contains the inverse distance wait between test and the train instance
            target_distance=1/distance[indices]
            
            '''creates the list containing the tuple of which one element is the class and the other is the inverse 
            distance between that particular class and the training instance'''
            
            middle_state=list(zip(target,target_distance))
           
            #final_state contains as much as k neighbors for prediction
            final_state=middle_state[0:k]
            
            '''A dictionary d is generated where the key is the class and the value is the 
            sum of inverse distance of each class within the k limit'''
            for a,b in final_state:
                if a not in d:
                    d[a]=b
                else:
                    d[a]+=b
            
            '''The key having the highest value is selected as the class of the query instance'''
            prediction+=[max(d, key=(lambda key: d[key]))]
        
                
        
                
      '''Returns the no of elementwise similarity between the predicted and the orginal numpy arrays''' 
      predicts=np.array(prediction)
      counts=np.count_nonzero(predicts==test_label)
      return (counts/1000)*100  


print('The accuracy for k =10 in distance weighted knn is {}'.format(Distance_Weighted_KNN(feature_train,feature_test,test_label,10,2)))  


#---------------------------------Part2 b-----------------------------------------

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

#_________________Calculate the accuracy for various values of k in standard and distance Weighted KNN______________

k_values=[]
accuracy_distance_weighted_knn=[]
accuracy_standard_knn=[]
for k in range(1,50):
    k_values+=[k]

    #print('the accuracy Distance Weighted KNN for k {} is {}'.format(k,Distance_Weighted_KNN(feature_train,feature_test,test_label,k,2)))
    '''accuracy_distnace_weighted_knn returns the accuracy values of distance weighted knn for each value of k by calling the distance weighted KNN function'''
    accuracy_distance_weighted_knn+=[Distance_Weighted_KNN(feature_train,feature_test,test_label,k,2)]          
    #print('the accuracy STANDARD KNN for k {} is {}'.format(k,Standard_KNN(feature_train,feature_test,test_label,k,2)))
    '''accuracy_standard_knn returns the accuracy values of standard knn for each values of k by calling the standard knn function'''
    accuracy_standard_knn+=[Standard_KNN(feature_train,feature_test,test_label,k,2)]
plt.title('Accuracy for distance_weighted_knn')
plt.xlabel('k_values')
plt.ylabel('Accuracy for different k values in Distance Weighted KNN')
plt.plot(k_values,accuracy_distance_weighted_knn,'r',label='Distance_weighted_knn')
plt.plot(k_values,accuracy_standard_knn,'b',label='standard_KNN')
plt.legend(loc='upper right')
plt.show()




    
#-----------Accuracy by using various distance matrix----------------------------

'''Here we vary p which is the power parameter for minkowski distance call the 
distance_weighted_knn and standard_knn functions that returns the accuracy for 
each value of p '''
diff_p=[]
a_standard=[]
a_weighted=[]
for p in range(1,10):
    diff_p+=[p]
    Accuracy_for_distance_weighted_knn=[]
    Accuracy_for_standard_knn=[]
    '''Here we compute the accuracy for both standard knn and distance weighted knn for k values from 1 to 9 and 
    find its average for a single value of p'''
    for k in range(1,10):
        Accuracy_for_distance_weighted_knn.append(Distance_Weighted_KNN(feature_train,feature_test,test_label,10,p))
        Accuracy_for_standard_knn.append(Standard_KNN(feature_train,feature_test,test_label,10,p))
    a_standard.append(np.mean(Accuracy_for_standard_knn))
    a_weighted.append(np.mean(Accuracy_for_distance_weighted_knn))

'''a_standard anda_weighted contains the accuarcy for each value of p'''

plt.title('Accuracy For DWKNN and KNN')
plt.xlabel('Different Value of p')     
plt.ylabel('Accuracy for different p')
plt.plot(diff_p,a_weighted,'r',label='DWKNN')
plt.plot(diff_p,a_standard,'b',label='KNN')
plt.legend(loc='lower right')
plt.show()


#-------------------lets do cross validation by dividing the data into 5 splits--------------------


accuracy_a=[]
accuracy_b=[]
total_data=np.vstack((data,data_test))
'''The entire data is divided into 5 equal parts'''
train=[total_data[0:1000,:],total_data[1000:2000,:],total_data[2000:3000,:],total_data[3000:4000,:],total_data[4000:5000,:]]
final_accuracy_a=[]
final_accuracy_b=[]
m=[]
K_FOLD=[1000,2000,3000,4000,5000]

for k in range(1,10):
    m+=[k]
    for k_fold in range(0,5000,1000):
        a=[]
        b=[]
        test=total_data[k_fold:k_fold+1000,:]
        '''For each of the section of the data act as the test set and accuracy is calculated for other sets'''
        for trainset in train:
            a+=[Distance_Weighted_KNN(trainset[:,:-1],test[:,:-1],test[:,-1],k,2)]
            b+=[Standard_KNN(trainset[:,:-1],test[:,:-1],test[:,-1],k,2)] 
        accuracy_a.append(max(a))
        accuracy_b.append(max(b))
    '''final_accuracy a and b contains the mean for each fold and is repeated for k neighbor times'''
    final_accuracy_a.append(np.mean(accuracy_a))
    final_accuracy_b.append(np.mean(accuracy_b))

print(final_accuracy_b)
plt.title('Accuracy for cross validation')
plt.xlabel('k_fold--splitted into 5')
plt.ylabel('Accuracy')
plt.plot(m,final_accuracy_a,'r',label='DWKNN')
plt.plot(m,final_accuracy_b,'b',label='KNN')
plt.legend(loc='upper right')
plt.show()
