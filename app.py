import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load the diabetes dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset\diabetes.csv")
    return df

df = load_data()

# Preprocess the data
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())

# Split data into features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# --- Model Training and Evaluation ---
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scale the data
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# 1. Baseline Model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sc, y_train)
y_pred = lr.predict(X_test_sc)

# 2. Adjusting Threshold
y_pred_proba = lr.predict_proba(X_test_sc)[:, 1]
y_pred_adjusted = (y_pred_proba > 0.3).astype(int)

# 3. Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=0
)
X_train_resampled_sc = sc.fit_transform(X_train_resampled)
X_test_resampled_sc = sc.transform(X_test_resampled)
lr_resampled = LogisticRegression(max_iter=1000)
lr_resampled.fit(X_train_resampled_sc, y_train_resampled)
y_pred_resampled = lr_resampled.predict(X_test_resampled_sc)

# 4. Class Weights
lr_weighted = LogisticRegression(class_weight={0: 1, 1: 3}, max_iter=1000)
lr_weighted.fit(X_train_sc, y_train)
y_pred_weighted = lr_weighted.predict(X_test_sc)

# 5. Hyperparameter Tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=param_grid, cv=5, scoring='recall')
grid_search.fit(X_train_sc, y_train)
best_lr = grid_search.best_estimator_
y_pred_tuned = best_lr.predict(X_test_sc)

# Prepare models and their predictions
models = {
    "Baseline Model": y_pred,
    "Adjusted Threshold (0.3)": y_pred_adjusted,
    "Oversampling": y_pred_resampled,
    "Class Weights": y_pred_weighted,
    "Hyperparameter Tuning": y_pred_tuned
}

# Calculate confusion matrices
conf_matrices = {}
for name, preds in models.items():
    if name == "Oversampling":
        actual = y_test_resampled
    else:
        actual = y_test
    conf_matrices[name] = confusion_matrix(actual, preds)

# --- Streamlit App ---
st.title("🩺 Diabetes Prediction App")

st.write("""
This app predicts the likelihood of a patient having diabetes based on various input features. 
It also demonstrates different techniques to improve model recall and displays confusion matrices for each technique.
""")

# User Input Features
st.sidebar.header("👩‍⚕️ Enter Patient Details")

def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 0)
    glucose = st.sidebar.slider("Glucose", 0, 300, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 150, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 1000, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Age", 21, 100, 30)
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader("👤 Patient Input Features")
st.write(input_df)

# Preprocess user input
input_scaled = sc.transform(input_df)

# --- Predictions for User Input ---
st.subheader("🔮 Prediction Results")

prediction_results = {}
for name, model in models.items():
    if name == "Oversampling":
        # Use the resampled scaler and model
        prediction = lr_resampled.predict(sc.transform(input_df))[0]
    elif name == "Adjusted Threshold (0.3)":
        prediction_proba = lr.predict_proba(sc.transform(input_df))[:, 1]
        prediction = int(prediction_proba > 0.3)
    elif name == "Hyperparameter Tuning":
        prediction = best_lr.predict(sc.transform(input_df))[0]
    elif name == "Class Weights":
        prediction = lr_weighted.predict(sc.transform(input_df))[0]
    else:
        prediction = lr.predict(sc.transform(input_df))[0]
    prediction_results[name] = "Diabetic" if prediction == 1 else "Non-Diabetic"

# Display predictions
for name, pred in prediction_results.items():
    st.write(f"**{name}** Prediction: {pred}")

# Display Confusion Matrices
st.subheader("📊 Confusion Matrices for Different Techniques")

for name, cm in conf_matrices.items():
    st.write(f"**{name}**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {name}')
    st.pyplot(fig)
    st.write("---")

# Display Evaluation Metrics
st.subheader("📈 Evaluation Metrics")

# Initialize a list to store metric dictionaries
metrics_list = []

for name, preds in models.items():
    if name == "Oversampling":
        actual = y_test_resampled
    else:
        actual = y_test
    acc = accuracy_score(actual, preds)
    prec = precision_score(actual, preds)
    rec = recall_score(actual, preds)
    f1 = f1_score(actual, preds)
    metrics_list.append({
        "Technique": name,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1 Score": round(f1, 2)
    })

# Create DataFrame from the list
metrics_df = pd.DataFrame(metrics_list)

st.dataframe(metrics_df.set_index("Technique"))

# Optional: Download the metrics as CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(metrics_df)

st.download_button(
    label="📥 Download Metrics as CSV",
    data=csv,
    file_name='evaluation_metrics.csv',
    mime='text/csv',
)

# Footer
st.markdown("""
---
*This app was developed to demonstrate diabetes prediction using various machine learning techniques to improve recall and evaluate their performance.*
""")
