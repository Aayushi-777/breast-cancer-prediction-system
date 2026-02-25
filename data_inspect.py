import pandas as pd
df = pd.read_csv("data.csv")
print("Shape: ", df.shape)
print("\nColumns:\n", df.columns)
print("\nRows:\n", df.head())
print("\nClass distribution:\n")
print(df.iloc[:,-1].value_counts())