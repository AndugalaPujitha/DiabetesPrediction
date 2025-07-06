import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis
#PIMA Diabetes Dataset
diabetes_dataset = pd.read_csv('C:/Users/Victus/Desktop/Diabetes/diabetes.csv')
diabetes_dataset.head()

#number of rows and columns in this dataset
diabetes_dataset.shape 

#getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

#0 --> Non - Diabetic
#1 --> Diabetic

diabetes_dataset.groupby('Outcome').mean()

#seperating the data and labels
X = diabetes_dataset.drop(columns= 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

#Data Standardization

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#TRAINING THE MODEL

classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier 
classifier.fit(X_train, Y_train)

#MODEL EVALUATION

#ACCURACY SCORE ON THE TRAINING DATA 
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

#Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy =  accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

#MAKING A PREDICTIVE SYSTEM

input_data = (4,110,92,0,0,37.6,0.191,30)
#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
    

















