import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# Load dataset
df = pd.read_csv("data.csv")
df = df.drop(columns=["id", "Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
X = df.drop(columns=["diagnosis"])
Y = df["diagnosis"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Load model
model = joblib.load("model.pkl")

#Get probabilities
probs = model.predict_proba(X_test)[:, 1]

#Compute curve
precision, recall, thresholds = precision_recall_curve(Y_test, probs)

#Plot
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()