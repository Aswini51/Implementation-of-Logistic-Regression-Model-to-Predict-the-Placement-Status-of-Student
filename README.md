# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: G ASWINI
RegisterNumber:  212224040037
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv("/content/Placement_Data.csv") 
data.head()
```

## Output:

![Screenshot 2025-04-14 140756](https://github.com/user-attachments/assets/b7865edf-ed30-40ff-9e3e-044c9111a04b)

## Program:
```
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

## Output:

![Screenshot 2025-04-14 140856](https://github.com/user-attachments/assets/cfbfcedb-57e7-4dc7-8822-254a70752b8f)

## Program:
```
data1.isnull()
```

## Output:

![Screenshot 2025-04-14 140954](https://github.com/user-attachments/assets/7527517a-7116-4e12-80a7-a02a2e63892e)

## Program:
```
data1.duplicated().sum()
```

## Output:

![Screenshot 2025-04-14 141006](https://github.com/user-attachments/assets/e613d69f-261a-48ef-8a29-2f391b877b8f)

## Program:
```
le = LabelEncoder()
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in cols:
    data1[col] = le.fit_transform(data1[col])
data1
```

## Output:

![Screenshot 2025-04-14 141027](https://github.com/user-attachments/assets/d526dde9-556b-49eb-9f38-63a98da2256c)

## Program:
```
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
```

## Output:

![Screenshot 2025-04-14 141035](https://github.com/user-attachments/assets/d547cb79-e965-445c-bf5a-196fd2118ebf)

## Program:
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
```

## Output:

![Screenshot 2025-04-14 141043](https://github.com/user-attachments/assets/24ab6823-1b10-4872-a627-02feede92c72)

## Program:
```
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

## Output:

![Screenshot 2025-04-14 141049](https://github.com/user-attachments/assets/badfe9f5-d819-4358-a9a6-e9eedda12507)

## Program:
```
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```

## Output:

![Screenshot 2025-04-14 141054](https://github.com/user-attachments/assets/2728c999-7f36-4fea-a5e9-c6ad086d6a4e)

## Program:
```
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

![Screenshot 2025-04-14 141228](https://github.com/user-attachments/assets/3382b90e-e9d1-40ce-809a-93c16c834b5d)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
