from LOAD_DATA import *
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convert X to a DataFrame using AttributeNames
X_df = pd.DataFrame(X[:, 1:], columns=attributeNames[1:])

# Add the target variable 'y' to the DataFrame
X_df["Rings"] = ring  # Assuming 'y' is already defined and matches the number of samples
print(y)
# Compute the correlation matrix including the target variable
corr_matrix = X_df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))  # Adjust figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=0, vmax=1)

# Set title
plt.title("Feature Correlation Heatmap", fontsize=20)
plt.savefig('Heatmap2.png')

# Show plot
plt.show()


