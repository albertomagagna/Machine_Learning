# LOAD DATA FOR ABALONE
import numpy as np
from ucimlrepo import fetch_ucirepo

# Fetch dataset from UCI Machine Learning Repository
abalone = fetch_ucirepo(id=1)

# Extract features and target from the dataset (as pandas dataframes)
features = abalone.data.features
target = abalone.data.targets

# Convert the feature dataframe to a NumPy array and exclude the first column
X_R = features.iloc[:].values
X_r = X_R[:, 1:].astype(float)

###########################################################################################
#%% Map sex categories to numerical values: M = -1, F = 1, I = 0
sex_letter = features.iloc[:, 0].values
sex_mapping = {'M': -1, 'F': 1, 'I': 0}
sex = np.array([sex_mapping[item] for item in sex_letter])

# Reshape the sex array and concatenate with the rest of the features
sex_reshaped = sex.reshape(-1, 1)
X = np.concatenate((sex_reshaped, X_r), axis=1)

# Extract attribute names from the features dataframe
attributeNames =  list(features.columns[:])
print(f'AttributeNames = {attributeNames}')

#%% #############################################################################
# sex_letter = features.iloc[:, 0].values
# sex_mapping = {'M': 0, 'F': 1, 'I': 2}
# sex = np.array([sex_mapping[item] for item in sex_letter])
# K = sex.max() + 1
# sex_encoding = np.zeros((sex.size, K))
# sex_encoding[np.arange(sex.size), sex] = 1
# X = np.concatenate((sex_encoding, X_r[:]), axis=1)
# # Extract attribute names from the features dataframe
# attributeNames = ['Sex_M', 'Sex_F', 'Sex_I'] + list(features.columns[1:])
# print(f'AttributeNames = {attributeNames}')
#%% #############################################################################

# Print the shape of the final feature matrix
print(f'X shape = {X.shape}')
print(f'X = {X}')

# Transform number of rings into class labels
# Classes: 'less than 9 rings', '9-10 rings range', 'more than 10 rings'
ring = target.iloc[:, 0].values
# classLabels = np.where(ring < 6, 'less than 6 rings', 
#               np.where(ring < 9, '6-8 rings', 
#               np.where(ring < 11, '9-10 rings', 
#               np.where(ring < 14, '11-13 rings', 'more than 13 rings'))))
print(ring)
classLabels = np.where(ring > 10, '2) rings range: > 10', np.where(ring < 9, '1) rings range: < 9', '3) rings range: 9-10'))
# Print the class labels
print(f'classLabels = {classLabels}')

# Extract unique class names and sort them
classNames = sorted(set(classLabels))
classNames = [str(name) for name in classNames]  # Convert np.str_ to regular str
print(f'classNames = {classNames}')

# Create a dictionary to map class names to numerical values
classDict = dict(zip(classNames, range(len(classNames))))
print(f'classDict = {classDict}')

# Convert class labels to numerical values using the class dictionary
y = np.array([classDict[value] for value in classLabels])
print(f'y = {y}')

# Compute the number of samples (N), number of attributes (M), and number of classes (C)
N = len(y)
M = len(attributeNames)
C = len(classNames)
print(f'N = {N}')
print(f'M = {M}')
print(f'C = {C}')

print("Ran Exercise 2.3.1 - loading the ABALONE data")
# %%
