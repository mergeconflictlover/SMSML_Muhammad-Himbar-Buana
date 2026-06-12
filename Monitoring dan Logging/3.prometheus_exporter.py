import time
import psutil
import requests
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik API Model
REQUEST_COUNT = Counter('himbarbuana_http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('himbarbuana_http_request_duration_seconds', 'HTTP Request Latency')
THROUGHPUT = Counter('himbarbuana_http_requests_throughput', 'Throughput request per detik')
FAILED_REQUESTS = Counter('himbarbuana_http_requests_failed_total', 'Total HTTP Requests Gagal')

# Metrik Prediksi Model
PREDICTION_CLASS_0 = Counter('himbarbuana_predictions_class_0_total', 'Total prediksi Non-Churn (0)')
PREDICTION_CLASS_1 = Counter('himbarbuana_predictions_class_1_total', 'Total prediksi Churn (1)')

# Metrik Sistem OS
CPU_USAGE = Gauge('himbarbuana_system_cpu_usage', 'Penggunaan CPU dalam persen')
RAM_USAGE = Gauge('himbarbuana_system_ram_usage', 'Penggunaan RAM dalam persen')
DISK_USAGE = Gauge('himbarbuana_system_disk_usage', 'Penggunaan Disk dalam persen')
NETWORK_SENT = Counter('himbarbuana_system_network_sent_bytes', 'Byte jaringan terkirim')
NETWORK_RECV = Counter('himbarbuana_system_network_recv_bytes', 'Byte jaringan diterima')
PROCESS_THREADS = Gauge('himbarbuana_system_process_threads', 'Jumlah thread aktif pada proses')

# Endpoint untuk scrape data metrik oleh Prometheus
@app.route('/metrics', methods=['GET'])
def metrics():
    # Update metrik OS real-time saat dipanggil oleh Prometheus scrape
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    
    net_io = psutil.net_io_counters()
    # Gunakan nilai delta atau total accumulative untuk counter
    NETWORK_SENT.inc(max(0, net_io.bytes_sent - NETWORK_SENT._value.get()))
    NETWORK_RECV.inc(max(0, net_io.bytes_recv - NETWORK_RECV._value.get()))
    
    proc = psutil.Process()
    PROCESS_THREADS.set(proc.num_threads())
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint wrapper untuk routing inference ke local mlflow model server
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()
    
    data = request.get_json()
    api_url = "http://127.0.0.1:5001/invocations"
    
    try:
        response = requests.post(api_url, json=data)
        
        if response.status_code != 200:
            FAILED_REQUESTS.inc()
            return jsonify({"error": f"Server returns status {response.status_code}"}), response.status_code
            
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        # Ekstrak hasil prediksi untuk metrik kelas
        predictions = response.json().get("predictions", [])
        for pred in predictions:
            if pred == 0:
                PREDICTION_CLASS_0.inc()
            elif pred == 1:
                PREDICTION_CLASS_1.inc()
                
        return jsonify(response.json())
        
    except Exception as e:
        FAILED_REQUESTS.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan di port 8001
    app.run(host='127.0.0.1', port=8001)
