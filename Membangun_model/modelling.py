import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_baseline():
    # Set run name
    mlflow.set_experiment("Telco Churn Prediction - himbarbuana")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    print("Membaca dataset preprocessed...")
    df = pd.read_csv("telco_customer_churn_preprocessing.csv")
    
    # Pisahkan X dan y
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Memulai run MLflow untuk baseline model...")
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        # Model Random Forest default
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Baseline Model: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_baseline()
