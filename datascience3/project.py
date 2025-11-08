import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
import pandas as pd
lucknow = pd.read_csv("Lucknow_1990_2022.csv")
print("Original Dataset:")
print(lucknow.head())
print("\n--- Min Max Scaler ---")
numeric_col = lucknow.select_dtypes(include='number').columns
scaler = MinMaxScaler()
chennai_normalized=pd.DataFrame(scaler.fit_transform(lucknow[numeric_col]),columns=numeric_col)
print(chennai_normalized.head())
print("\n--- Standard Scaler ---")
numeric_col1 = lucknow.select_dtypes(include='number').columns
scaler =StandardScaler()
chennai_standardized=pd.DataFrame(scaler.fit_transform(lucknow[numeric_col1]),columns=numeric_col1)
print(chennai_standardized.head())
plt.figure(figsize=(8,6))
plt.hist(lucknow['tavg'], bins=10, edgecolor='black', color='pink')
plt.title("Distribution of Average Temperature (tavg)")
plt.xlabel("Average Temperature (°C)")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()