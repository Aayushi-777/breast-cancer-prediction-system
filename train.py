import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

#Load dataset
df = pd.read_csv("data.csv")

#Drop useless columns and clean column names
df = df.drop(columns=["id","Unnamed: 32"])
df.columns = df.columns.str.replace(" ", "_")

#Convert diagnosis to numeric
df["diagnosis"] = df["diagnosis"].map({"M":1, "B":0})

#Separate features and target
X = df.drop(columns=["diagnosis"])
Y = df["diagnosis"]

#Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)

#Save Model
joblib.dump(model, "model.pkl")

#Evaluate
Y_pred = model.predict(X_test)
print("Model Trained Successfully")