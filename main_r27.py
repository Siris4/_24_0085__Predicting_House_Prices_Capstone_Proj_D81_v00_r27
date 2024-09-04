import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Step 1: Visualize the original distribution of PRICE
plt.figure(figsize=(10, 6))
sns.displot(data['PRICE'], kde=True, height=6, aspect=1.5, bins=30)
plt.title('Original Distribution of PRICE')
plt.xlabel('PRICE')
plt.ylabel('Density')
plt.grid(True)
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\original_price_distribution.png')

# Step 2: Apply the log transformation to PRICE
log_price = np.log(data['PRICE'])

# Step 3: Visualize the log-transformed distribution of PRICE
plt.figure(figsize=(10, 6))
sns.displot(log_price, kde=True, height=6, aspect=1.5, bins=30)
plt.title('Log-Transformed Distribution of PRICE')
plt.xlabel('Log(PRICE)')
plt.ylabel('Density')
plt.grid(True)
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\log_price_distribution.png')

# Step 4: Calculate skewness before and after the log transformation
original_skewness = skew(data['PRICE'])
log_skewness = skew(log_price)

# Output the skewness values
print(f"Skewness of original PRICE: {original_skewness:.4f}")
print(f"Skewness of log-transformed PRICE: {log_skewness:.4f}")
