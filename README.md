# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3. Import LabelEncoder and encode the dataset.

4. Import LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array.

6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7. Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Thirukaalathessvarar S
RegisterNumber: 212222230161
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placement_data:
![image](https://github.com/MOHAMED-FAREED-22001617/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121412904/fe66412d-9902-4f22-81e9-ac7ed05a6b23)
### Salary_data:
![ml_exp2_1](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/6b8a6b39-6a17-4393-a511-19d3f6ef4c3b)
### ISNULL():
![ml_exp2_2](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/5f3ed024-8783-434c-adf9-e64cb1c69d6e)
### DUPLICATED():
![ml_exp2_3](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/e96b3cd8-2a80-415b-b34b-e0e9a48e32c5)
### Print Data:
![ml_exp2_4](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/9705756a-df03-4289-98d5-b78271d9d195)
### iloc[:,:-1]:
![ml_exp2_5](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/499ae47e-32aa-4ec5-afba-35d6c20a5112)
### Data_Status:
![ml_exp2_6](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/1f6fe82e-9b1d-4591-a049-5aceea3655ba)
### Y_Prediction array:
![ml_exp2_7](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/a7cce56f-0932-4ceb-ba1a-377311a77476)
### Accuray value:
![ml_exp2_8](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/2c32c8b5-3195-4428-a766-72047862a62d)
### Confusion Array:
![ml_exp2_9](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/c651841c-fd25-40a5-92d0-ed185f772c0f)
### Classification report:
![ml_exp2_10](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/04affd4a-4973-46d9-976c-eca15e22115b)
### Prediction of LR:
![ml_exp2_11](https://github.com/Thirukaalathessvarar-S/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121166390/1e8c27ed-0077-4d4c-9c8b-57419c5aa96d)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
