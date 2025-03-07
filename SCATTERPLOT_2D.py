# Exercise 2.3.5
# (requires data from exercise 2.3.1)
from LOAD_DATA import *
import matplotlib.pyplot as plt


outlier_mask = (X[:, 3] > 0.5) # limit in Height of the Abalone to 0.4*200 = 80 mm = 8cm
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask, :]
y = y[valid_mask]
N = len(y)

plt.figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        if m1 == m2:
            plt.hist(X[:, m1], bins=20, color='gray')
        else:
            for c in range(C):
                class_mask = y == c
                plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".", label=classNames[c])
        if m1 == M - 1:
            plt.xlabel(attributeNames[m2], rotation=0)
        else:
            plt.xticks([])
        if m2 == 0:
            plt.ylabel(attributeNames[m1], rotation=30, loc='top')
        else:
            plt.yticks([])
plt.subplots_adjust(right=0.85)
plt.figlegend(['histogram data'] + classNames, loc='lower right', bbox_to_anchor=(0.99, 0.1))
plt.savefig('SCATTERPLOT_2D.png')
plt.show()

print("Ran Exercise 2.3.5")
