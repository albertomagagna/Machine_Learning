# Exercise 2.3.2
import numpy as np
# (requires data from exercise 2.3.1 so will run that script first)
from LOAD_DATA import *
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
print(X)
print(range(M))
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i],bins=20, color='navy')
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N/1.5)
    if i % int(v) == 0:
        plt.ylabel('Frequency')
    else:
        plt.gca().set_yticklabels([])
    
plt.suptitle("Abalone: Histogram", fontsize=20)
plt.savefig('Histogram_WITH_outliers.png')
plt.show()

print("Ran Exercise 2.3.2")
