# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VINODINI R
RegisterNumber:  212223040244
*/


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data = pd.read_csv("Placement_Data.csv")

# Print the entire DataFrame
print("Placement Data:")
print(data)

# Print only the salary column (if it exists)
if 'salary' in data.columns:
    print("\nSalary Data:")
    print(data['salary'])
else:
    print("\n'Salary' column not found in DataFrame")

# Remove unnecessary columns (if any)
data1 = data.drop(["salary"], axis=1, errors='ignore')

# Check for missing values
print("\nMissing Values Check:")
print(data1.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows Check:")
print(data1.duplicated().sum())

# Print the cleaned data
print("\nCleaned Data:")
print(data1)

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
categorical_columns = ['workex', 'status', 'hsc_s']  # List of categorical columns to encode
for column in categorical_columns:
    if column in data1.columns:
        data1[column] = le.fit_transform(data1[column])
    else:
        print(f"'{column}' column not found in DataFrame")

# Prepare features and target
x = data1.drop('status', axis=1, errors='ignore')  # Features
y = data1['status']  # Target

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train the model
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_report1)

# Print the y_pred array
print("\nY Prediction Array:")
print(y_pred)

```

## Output:
## Placement Data :
![image](https://github.com/user-attachments/assets/f11b4bad-9971-4743-8a7d-805395fd1651)
## Salary Data :
![image](https://github.com/user-attachments/assets/d7a9bca2-ba40-4fc2-9b15-52e88e7aa6b0)
## Checking the null() function :
![image](https://github.com/user-attachments/assets/83abc147-e2f1-4af2-8188-ad69f412fa54)
## Data Duplicate :
![image](https://github.com/user-attachments/assets/63dda5bb-c0d2-4042-bb0c-fd805e98e3a2)
## Clean Data :
![image](https://github.com/user-attachments/assets/c05455b7-4279-4ac3-b095-6ee20f330d9a)
## Y-Prediction Array :
![image](https://github.com/user-attachments/assets/f7c25ad4-5e81-4d98-bafc-adb95e90bc7a)
## Missing Values Check :
![image](https://github.com/user-attachments/assets/fb9c5782-0b08-4953-9149-5132b25540a5)
## Accuracy value :
![image](https://github.com/user-attachments/assets/4f0e9014-3864-4456-aa48-ab2a1324eddd)
## Confusion array :
![image](https://github.com/user-attachments/assets/d7e0f1c4-a447-413e-86b9-8c3722c22df5)
## Classification Report :
![image](https://github.com/user-attachments/assets/f9dfce0b-608d-4aed-8de9-9fbbe28e96ba)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
