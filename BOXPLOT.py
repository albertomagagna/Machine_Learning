# Exercise 2.3.2
import numpy as np
# (requires data from exercise 2.3.1 so will run that script first)
from LOAD_DATA import *
import matplotlib.pyplot as plt
from scipy.stats import zscore

print(X[:])
plt.figure(figsize=(8, 7))
plt.boxplot(zscore(X[:, 1:], ddof=1))  # Exclude the first column (Sex) for zscore
plt.xticks(range(1, M), attributeNames[1:], rotation=30)  # Exclude the first column name (Sex)
plt.ylabel("Normalized feature values")
plt.title("Abalone data set - boxplot", fontsize=20)

plt.savefig('Boxplot with outliers.png')
plt.show()
print("Ran Exercise 2.3.3")