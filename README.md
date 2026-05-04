## 🩺 Breast Cancer Prediction System

A Machine Learning based web application that predicts whether a tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using clinical features from the Breast Cancer dataset from Kaggle.
Built using **Scikit-learn, FastAPI and Bootstrap**, this project demonstrates with an interactive web interface.

---

## 🚀 Features
- 🔬 Predicts tumor classification (Malignant / Benign)
- 📊 Displays probability of malignancy
- 🌐 FastAPI backend
- 🎨 Responsive Bootstrap-based frontend
- ⚡ Real-time predictions
- 🧠 Machine Learning powered (Random Forest / Logistic Regression)

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data
- Breast Cancer dataset (30 numerical features)
- Features include:
  - Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness
  - Compactness
  - Concavity
  - Symmetry
  - Fractal dimension

---

### 2️⃣ Model Training
- Data preprocessing
- Train-test split
- Model training using:
  - Random Forest Classifier (or your chosen model)
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

### 3️⃣ Model Saving
- Model saved using:
joblib
- Loaded inside FastAPI for real-time inference

---

### 🏗️ Project Structure
breast_cancer_project/
│
├── static/
│ └── icon.png
│
├── templates/
│ └── index.html
│
├── model.pkl
├── app.py
├── requirements.txt
└── README.md

---

## Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Aayushi-777/breast-cancer-prediction-system.git
cd breast-cancer-prediction-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the Application

```bash
uvicorn app:app --reload --port 8000
```
And open the link "https://127.0.0.1:8000"

## 🧩 Technologies used

python
Scikit-learn
FastAPI
Pandas
NumPy
Joblib
Jinja2
Bootstrap 5
