
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv(r"/Users/shashi/Desktop/Data (1).csv")

# 3. SEPARATE INDEPENDENT AND DEPENDENT VARIABLES

X = dataset.iloc[:, :-1].values  

# X: Independent variables (all columns except the last)

y = dataset.iloc[:, 3].values  

# y: Dependent variable (4th column, index starts from 0)


# 4. HANDLE MISSING VALUES IN NUMERICAL DATA

from sklearn.impute import from sklearn.impute import SimpleImputer
import numpy as np
# # Tool to fill in missing values


imputer = SimpleImputer()  
# Create the imputer object with default strategy (mean)

imputer = imputer.fit(X[:, 1:3])  
# Fit the imputer to columns 2 and 3 (index 1 and 2)


X[:, 1:3] = imputer.transform(X[:, 1:3])  
# Replace missing values in X with the mean of each column

print(X[:, 1:3])

# ------------------------------------------------------
# 5. ENCODE CATEGORICAL DATA (INDEPENDENT VARIABLE)
# ------------------------------------------------------


## LabelEncoder takes unique text labels (like country names: 'India', 'France', 'Germany') and converts them into numeric values starting from 0.


from sklearn.preprocessing import LabelEncoder 


labelencoder_X= LabelEncoder()
# Create label encoder object for independent variable


X[:,0] = labelencoder_X.fit_transform(X[:,0])
# Convert first column (e.g., country names) to numerical codes

print(X[:,0]) 
# use print function to check if it has been  encoded 


# ------------------------------------------------------
# 6. ENCODE CATEGORICAL DATA (DEPENDENT VARIABLE)
# ------------------------------------------------------

labelencoder_y = LabelEncoder()  
# Create label encoder for dependent variable

y = labelencoder_y.fit_transform(y)  
# Convert y values (e.g., 'Yes', 'No') to 0s and 1s

# ------------------------------------------------------
# 7. SPLIT THE DATASET INTO TRAINING AND TEST SETS
# ------------------------------------------------------


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
# Split data into training and testing sets (80%-20%)
# random_state ensures reproducibility - everytime you execute it gives the same value instead of random values 


# 8. FEATURE SCALING (OPTIONAL BUT RECOMMENDED)
# ------------------------------------------------------

from sklearn.preprocessing import Normalizer  

# Normalizer scales each row (sample) to unit norm, meaning it transforms the values in each row so that the total length (magnitude) becomes 1.

sc_X = Normalizer()  
# Create a normalizer object (scales input vectors to unit norm)

X_train = sc_X.fit_transform(X_train)  
# Fit to training data and transform it

X_test = sc_X.transform(X_test)  
# Transform the test data using same scale

















