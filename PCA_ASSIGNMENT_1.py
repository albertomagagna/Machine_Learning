#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import zscore
import importlib_resources
import xlrd
from ucimlrepo import fetch_ucirepo 
from scipy.linalg import svd
  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
feature = abalone.data.features 
target = abalone.data.targets 

########## transform sex into 1-out-of-K encoding  ###########

X_r = feature.iloc[:, 1:].values.astype(float) # Exclude first column

# M = -1, F = 1, I = 0
sex_letter = feature.iloc[:, 0].values
sex_mapping = {'M': -1, 'F': 1, 'I': 0}
sex = np.array([sex_mapping[item] for item in sex_letter])
print(sex.shape)

#K = sex.max() + 1
# sex_encoding = np.zeros((sex.size, K))
# sex_encoding[np.arange(sex.size), sex] = 1
# X = np.concatenate((sex_encoding, X_r[:, :-1]), axis=1)
sex_reshaped = sex.reshape(-1, 1)
X = np.concatenate((sex_reshaped, X_r), axis=1)

print(f'X shape = ', X.shape)

############ transform number of rings into set ##################################
# ring < 9, 9 <=ring <= 10, ring > 10
ring = target.iloc[:, 0].values
classLabels = np.where(ring > 10, '2) rings range: > 10', np.where(ring < 9, '1) rings range: < 9', '3) rings range: 9-10'))
# Print the class labels
print(f'classLabels = {classLabels}')


# Extract attribute names
attributeNames = list(feature.columns)
print(f'AttributeNames = ', attributeNames)

# Extract class names to python list,
# then encode with integers (dict)
# classLabels = target.iloc[:,0].values
# print(f'classLabels = ', classLabels)

classNames = sorted(set(classLabels))
classNames = [str(name) for name in classNames]  # Convert np.str_ to regular str
print(f'classNames = ', classNames)

classDict = dict(zip(classNames, range(len(classNames))))
print(f'classDict = ', classDict)

# Extract vector y, convert to NumPy matrix and transpose
y = np.array([classDict[value] for value in classLabels])
print(f'y = ', y)

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
print(f'N = ', N)
print(f'M = ', M)
print(f'C = ', C)

print("Ran Exercise 2.3.1 - loading the ABALONE data")

# We start with a box plot of each attribute
plt.figure()
plt.title("Abalone: Boxplot")
plt.boxplot(X)
plt.xticks(range(1, M ), attributeNames[:-1], rotation=45)

# From this it is clear that there are some outliers in the Abalone
# attribute 
# However, it is impossible to see the distribution of the data, because
# the axis is dominated by these extreme outliers. To avoid this, we plot a
# box plot of standardized data (using the zscore function).
plt.figure(figsize=(12, 6))
plt.title("Abalone: Boxplot (standarized)")
plt.boxplot(zscore(X[:,1:], ddof=1))
plt.xticks(range(1, M ), attributeNames[1:], rotation=45) #!!!!!!!!!!!! modified range and attributes

# This plot reveals that there are clearly some outliers in Heigth attribute, i.e. attribute number 3
plt.show()

# Next, we plot histograms of all attributes.
plt.figure(figsize=(14, 9))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N)  # Make the y-axes equal for improved readability
    if i % v != 0:
        plt.yticks([])
    if i == 0:
        plt.title("Abalone: Histogram")

plt.show()

# This confirms our belief about outliers in attribute Height.
# To take a closer look at this, we next plot histograms of the
# attributes we suspect contains outliers
plt.figure(figsize=(14, 9))
m = [3]
for i in range(len(m)):
    plt.subplot(1, len(m), i + 1)
    plt.hist(X[:, m[i]], 100)
    plt.xlabel(attributeNames[m[i]])
    plt.ylim(0, N/10)  # Make the y-axes equal for improved readability
    if i > 0:
        plt.yticks([])
    if i == 0:
        plt.title("Abalone: Histogram (selected attributes)")

plt.show()

# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.

####ANALYSIS OF FULL DATASET CONTAINING OUTLIERS####
mean_x = X.mean(axis=0)
std_x = X.std(axis=0,ddof=1) # ddof: Delta Degrees of freedom
median_x = np.median(X,axis=0)

# Compute mean, standard deviation, and median for the attribute rings
mean_rings = ring.mean()
std_rings = ring.std(ddof=1)
median_rings = np.median(ring)

print(f'Rings: Mean = {mean_rings}, Standard Deviation = {std_rings}, Median = {median_rings}')
# Count the number of males, females, and infants
num_males = np.sum(sex == -1)
num_females = np.sum(sex == 1)
num_infants = np.sum(sex == 0)

print(f'Number of males: {num_males}')
print(f'Number of females: {num_females}')
print(f'Number of infants: {num_infants}')
# Print mean, standard deviation, and median with attribute names
for attr, mean, std, median in zip(attributeNames, mean_x, std_x, median_x):
    print(f'{attr}: Mean = {mean}, Standard Deviation = {std}, Median = {median}')
print("Mean:", mean_x)
print("Standard Deviation:", std_x)
print("Median:", median_x)


#%%

outlier_mask = (X[:, 3] > 0.4)  # cosider height higher than 80mm as outlier
valid_mask = np.logical_not(outlier_mask)

#Analysing the two outliers found for the "Height" attribute, we decide to eliminate them
#as their value for height does not seem to be physical, after a research on typival values for abalone's height
# Finally we will remove these from the data set
X = X[valid_mask, :]
y = y[valid_mask]
N = len(y)

# Now, we can repeat the process to see if there are any more outliers
# present in the data. We take a look at a histogram of all attributes:
plt.figure(figsize=(14, 9))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N)  # Make the y-axes equal for improved readability
    if i % v != 0:
        plt.yticks([])
    if i == 0:
        plt.title("ABALONE: Histogram (after outlier detection)")

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

plt.show()

#### RECOMPUTE STATISTICS ON THE DATASET WITHOUT OUTLIERS ####
mean_x_no_ol = X.mean(axis=0)
std_x_no_ol = X.std(axis=0,ddof=1) # ddof: Delta Degrees of freedom
median_x_no_ol = np.median(X,axis=0)

print("Mean without outliers:", mean_x_no_ol)
print("Standard Deviation without outliers:", std_x_no_ol)
print("Median without outliers:", median_x_no_ol)

print("Ran Exercise 2.4.1")

# %%
#### PERFORMING PCA ANALYSIS ####

# Subtracting the mean from the data matrix

X_0_mean = X -mean_x_no_ol
X_0_mean = X_0_mean / np.std(X_0_mean, 0)

# Singular Value Decomposition
U, S, Vh = svd(X_0_mean)
V = Vh.T

# Variance explained
rho = (S * S) / (S * S).sum()

threshold = 0.95

# Plot variance explained

plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

# %%
# Finding projection of data onto PC space
Z = X_0_mean @ V

# Indices of the principal components to be plotted
i = 0
j = 1
# Plot PCA of the data
f = plt.figure()
plt.title("Abalone data: PCA")
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))

# Output result to screen
plt.show()


# %%
pcs = [0, 1, 2]
legendStrs = ["PC" + str(e + 1) for e in pcs]
bw = 0.2
r = np.arange(1, M + 1)

for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)

plt.xticks(r + bw, attributeNames, rotation=45)  # Rotate x-axis labels by 45 degrees
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("Abalone: PCA Component Coefficients")
plt.show()

#%%
#Inspecting the plot, we notice that the first PC has e predominant role
#and it is given almost completely by the sex. So, since we are looking for 
#explanation also coming from other features we decide to remove this variable from the analysis
print("PC1:")
print(V[:, 0].T)

print("PC2:")
print(V[:, 1].T)

#%%
# Remove the "sex" attribute (first column) from the data matrix
X_new = X[:, 1:]

# Recompute the mean and subtract it from the new data matrix
mean_x_new = X_new.mean(axis=0)
X_new_0_mean = X_new - mean_x_new
X_new_0_mean = X_new_0_mean / np.std(X_new_0_mean, 0)

# Perform Singular Value Decomposition (SVD) on the new data matrix
U, S, Vh = svd(X_new_0_mean)
V = Vh.T

# Variance explained by the new principal components
rho = (S * S) / (S * S).sum()

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [0.95, 0.95], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

# Finding projection of data onto PC space
Z_new = X_new_0_mean @ V

# Indices of the principal components to be plotted
i = 0
j = 1
# Plot PCA of the data
plt.figure()
plt.title("Abalone data: PCA")
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z_new[class_mask, i], Z_new[class_mask, j], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC{0}".format(i + 1))
plt.ylabel("PC{0}".format(j + 1))
plt.show()

# Plot PCA component coefficients
pcs = [0, 1, 2]
legendStrs = ["PC" + str(e + 1) for e in pcs]
bw = 0.2
r = np.arange(1, M)  # Adjusted range to match the new number of attributes

for i in pcs:
    plt.bar(r + i * bw, V[:, i], width=bw)

plt.xticks(r + bw, attributeNames[1:], rotation=45)  # Rotate x-axis labels by 45 degrees
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend(legendStrs)
plt.grid()
plt.title("Abalone: PCA Component Coefficients")
plt.show()

print("PC1:")
print(V[:, 0].T)


#%%
# Plot PCA of the data (first two principal components)
plt.figure()
plt.title("Abalone data: PCA (without sex attribute)")
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z_new[class_mask, 0], Z_new[class_mask, 1], "o", alpha=0.5)
plt.legend(classNames)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%

# Perform Singular Value Decomposition (SVD) on the new data matrix
U, S, Vh = svd(X_new_0_mean)
V = Vh.T

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot attribute coefficients in principal component space
plt.figure()
for att in range(V.shape[1]):
    plt.arrow(0, 0, V[att, i], V[att, j])
    plt.text(V[att, i], V[att, j], attributeNames[att + 1])  # Adjusted index to match the new attributes
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel("PC" + str(i + 1))
plt.ylabel("PC" + str(j + 1))
plt.grid()
# Add a unit circle
plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01)))
plt.title("Abalone data: Attribute coefficients in PC space")
plt.axis("equal")
plt.show()