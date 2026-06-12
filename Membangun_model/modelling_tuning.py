import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    PrecisionRecallDisplay
)
import mlflow
import mlflow.sklearn
import dagshub

import socket

def run_tuning():
    # 1. Integrasi DagsHub (Advance Level)
    # Kami menggunakan environment variable atau fallback otomatis jika dijalankan offline
    dagshub_owner = os.getenv("DAGSHUB_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO")
    
    if dagshub_owner and dagshub_repo:
        try:
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
            print(f"Berhasil menginisialisasi online tracking DagsHub: {dagshub_owner}/{dagshub_repo}")
        except Exception as e:
            print(f"Gagal inisialisasi DagsHub: {e}. Menggunakan MLflow lokal.")
    else:
        print("Variabel lingkungan DagsHub tidak diset. Menggunakan MLflow lokal.")
        # Setup local tracking URI jika port 5000 aktif
        a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        a_socket.settimeout(1.0)
        location = ("127.0.0.1", 5000)
        result_of_check = a_socket.connect_ex(location)
        a_socket.close()
        if result_of_check == 0:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            print("Server tracking MLflow terdeteksi aktif pada http://127.0.0.1:5000")
        else:
            print("Server tracking MLflow lokal tidak aktif. Logging langsung ke direktori ./mlruns")

    mlflow.set_experiment("Telco Churn Prediction - himbarbuana")

    # 2. Muat Data
    print("Membaca dataset preprocessed...")
    df = pd.read_csv("telco_customer_churn_preprocessing.csv")
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Hyperparameter Tuning
    print("Memulai Hyperparameter Tuning dengan GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1'
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Hyperparameter terbaik: {best_params}")
    
    # Evaluasi Model Terbaik
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Metrik Evaluasi:\n- Accuracy: {acc:.4f}\n- Precision: {prec:.4f}\n- Recall: {rec:.4f}\n- F1-Score: {f1:.4f}")
    
    # 4. Manual Logging ke MLflow (Skilled & Advance)
    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        # Log parameter terbaik secara manual
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)
            
        # Log parameter pencarian lainnya
        mlflow.log_param("tuning_cv", 3)
        mlflow.log_param("tuning_scoring", "f1")
        
        # Log metrik secara manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Save model secara lokal terlebih dahulu untuk menghasilkan MLmodel, conda.yaml, model.pkl, dll.
        import shutil
        temp_model_dir = "temp_model_save"
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

        mlflow.sklearn.save_model(best_model, temp_model_dir)

        # Log folder tersebut sebagai artifacts ke path "model" di DagsHub
        mlflow.log_artifacts(temp_model_dir, artifact_path="model")

        # Bersihkan folder temp lokal
        shutil.rmtree(temp_model_dir)
        
        # 5. Pembuatan dan Logging Artefak Tambahan (Kriteria Advance)
        print("Membuat dan merekam artefak tambahan...")
        
        # Artefak 1: Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix - Telco Churn')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        print(f"Artefak {cm_path} berhasil disimpan ke MLflow.")
        
        # Artefak 2: Plot Precision-Recall Curve
        plt.figure(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, name="Random Forest")
        plt.title('Precision-Recall Curve - Telco Churn')
        plt.tight_layout()
        
        pr_path = "precision_recall_curve.png"
        plt.savefig(pr_path)
        plt.close()
        mlflow.log_artifact(pr_path)
        print(f"Artefak {pr_path} berhasil disimpan ke MLflow.")
        
        # Artefak 3: Teks Laporan Klasifikasi (Classification Report)
        report_text = classification_report(y_test, y_pred)
        report_path = "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        mlflow.log_artifact(report_path)
        print(f"Artefak {report_path} berhasil disimpan ke MLflow.")
        
        # Bersihkan berkas lokal sementara setelah direkam
        for temp_file in [cm_path, pr_path, report_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    print("Seluruh proses tuning dan logging manual selesai!")

if __name__ == "__main__":
    run_tuning()
