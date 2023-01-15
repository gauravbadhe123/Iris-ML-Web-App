
import pandas as pd  # data preprocessing
# pandas is aliased as pd 
import numpy as np  # mathematical computation
# numpy is aliased as np
from sklearn.model_selection import train_test_split
# train_test_split to split data into train and test

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('iris.csv')
# print(df.shape)           # 145,5
# print(df.isnull().sum())  # there are no null values
# print(df.columns)         # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
# print(df.head(4))          
# Target variable - label
# value counts for label
# print(df['label'].value_counts())  # setosa - 49, versicolor - 50 , virginica - 50

# Select the dependent and independent features
x = df.iloc[:,:-1]
y = df['label']

# print(type(x))   # DataFrame
# print(type(y))   # Series
# print(x.shape)   # (149,4)
# print(y.shape)   # (149,)

# Split the data into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

# print(x_train.shape)  # (111,4)
# print(x_test.shape)   # (38,4)
# print(y_train.shape)  # (111,)
# print(y_test.shape)   # (38,)

# Train the ML Model
lr_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_split=15)
rf_model = RandomForestClassifier(n_estimators=70,criterion='gini',
                                    max_depth=4,min_samples_split=15)

lr_model.fit(x_train,y_train)
dt_model.fit(x_train,y_train)
rf_model.fit(x_train,y_train)

# Save the trained ML Model
# Saving the ml model in binary format
pickle.dump(lr_model,open('lr_model.pkl','wb'))
pickle.dump(dt_model,open('dt_model.pkl','wb'))
pickle.dump(rf_model,open('rf_model.pkl','wb'))

# wb - write binary

# Terminal commands
# To run a Python file:  pythton abc.py
# To stop teh Server: Ctrl + Cs



