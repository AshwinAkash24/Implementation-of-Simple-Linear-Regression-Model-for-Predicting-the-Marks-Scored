# EX-02:Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Collection and Exploration: Load and explore the dataset using pd.read_csv() and df.head(), df.tail(), df.info().<br>
2.Data Preparation: Extract features (X) and target (y) from the dataset using df.iloc[:,:-1].values and df.iloc[:,-1].values.<br>
3.Data Splitting: Split the data into training and testing sets with train_test_split(x, y, test_size=1/3, random_state=1).<br>
4.Model Training: Train the linear regression model using LinearRegression() and reg.fit(x_train, y_train).<br>
5.Model Evaluation: Predict the values with reg.predict(x_test) and evaluate performance using MSE, MAE, and RMSE.<br>
6.Visualization: Plot the actual vs predicted values with plt.scatter() and the regression line using plt.plot().<br>

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ashwin Akash M
RegisterNumber:  212223230024
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
df.info()
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=1)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
mse=mean_squared_error(y_test,y_pred)
print("MSE =",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE =",mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,y_pred)
plt.title("Test of H VS S")
plt.xlabel("Scores")
plt.ylabel("Hours")
```

## Output:
![image](https://github.com/user-attachments/assets/9afee5ed-b1de-4ebb-96a5-67935d4176bc)<br>
![image](https://github.com/user-attachments/assets/5b53f535-c067-4ec0-99c0-ebb9cd0494c8)<br>
![image](https://github.com/user-attachments/assets/139448b9-c92f-4c7a-820c-2879988ef765)<br>
![image](https://github.com/user-attachments/assets/f6e53253-09e4-4a41-b523-e9ad1abc37a7)<br>
![image](https://github.com/user-attachments/assets/9b31cbb0-9cec-486f-929a-ac45d018f932)<br>

![image](https://github.com/user-attachments/assets/a2c5bf93-3c08-4bc1-9699-ef6d29e26aae)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
