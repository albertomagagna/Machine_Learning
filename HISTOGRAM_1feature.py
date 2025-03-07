# Exercise 2.3.2
import numpy as np
# (requires data from exercise 2.3.1 so will run that script first)
from LOAD_DATA import *
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))
i =3
plt.hist(X[:, i],bins=100, color='navy')
plt.xlabel(attributeNames[i])
plt.ylim(0, N/7)
plt.title("Abalone: Histogram", fontsize=20)

plt.savefig('ex2_3_2_HISTOGRAM_1feature.png')
plt.show()
print("Ran Exercise 2.3.2")
