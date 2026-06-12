import time
import argparse
import pandas as pd
import requests

def send_traffic(num_requests, delay, trigger_errors):
    print("Membaca dataset preprocessed untuk sample...")
    try:
        df = pd.read_csv("telco_customer_churn_preprocessing.csv")
    except FileNotFoundError:
        df = pd.read_csv("telco_customer_churn_preprocessing")
        
    X = df.drop(columns=["Churn"])
    
    # URL Exporter Flask Himbar
    url = "http://127.0.0.1:8001/predict"
    
    print(f"Memulai pengiriman {num_requests} request ke {url}...")
    for i in range(num_requests):
        # Ambil baris data secara random
        sample = X.sample(n=1)
        
        # Susun format payload dataframe mlflow split
        payload = {
            "dataframe_split": {
                "columns": list(sample.columns),
                "data": sample.values.tolist()
            }
        }
        
        # Pemicu error sengaja jika flag diset aktif
        if trigger_errors and i % 5 == 0:
            payload = {"invalid_format": {}}
            print(f"[{i+1}/{num_requests}] Mengirim request error sengaja...")
        else:
            print(f"[{i+1}/{num_requests}] Mengirim request inference sukses...")
            
        try:
            res = requests.post(url, json=payload)
            print(f"Respons: Code {res.status_code} - {res.json()}")
        except Exception as e:
            print(f"Gagal menghubungi exporter: {e}")
            
        time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_requests", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--trigger_errors", action="store_true")
    args = parser.parse_args()
    
    send_traffic(args.num_requests, args.delay, args.trigger_errors)
