# Submission Proyek Akhir: Membangun Sistem Machine Learning (MLOps)

Repositori ini berisi berkas-berkas submission lengkap untuk proyek akhir MLOps Dicoding milik **Muhammad Himbar Buana** (username Dicoding: `himbarbuana`, username GitHub: `mergeconflictlover`).

Proyek ini telah memenuhi seluruh kriteria penilaian hingga tingkat **Advanced (Bintang 5)**.

## 📂 Struktur Berkas Submission

```text
├── Eksperimen_SML_Muhammad-Himbar-Buana.txt   # Tautan repositori eksperimen model
├── Workflow-CI.txt                            # Tautan repositori workflow retraining CI/CD
├── Membangun_model
│   ├── modelling.py                           # Kode latih baseline model (Autolog)
│   ├── modelling_tuning.py                    # Kode hyperparameter tuning & manual logging
│   ├── requirements.txt                       # Dependensi pustaka python model
│   ├── DagsHub.txt                            # Tautan DagsHub Experiment Tracking
│   ├── docker_hub.txt                         # Tautan repositori Docker Hub image
│   ├── telco_customer_churn_preprocessing.csv # Dataset bersih siap latih
│   ├── screenshoot_dashboard.jpg              # Bukti pencatatan dashboard MLflow/DagsHub
│   ├── screenshoot_artifak.jpg                # Bukti pencatatan artefak model MLflow/DagsHub
│   └── screenshoot_docker_hub.jpg             # Bukti build image Docker ter-upload
└── Monitoring dan Logging
    ├── 1.bukti_serving/                       # Tangkapan layar Flask API model serving
    ├── 2.prometheus.yml                       # Konfigurasi target scraper Prometheus (port 8001)
    ├── 3.prometheus_exporter.py               # Kode wrapper API Flask + Prometheus Client (11 metrik)
    ├── 4.bukti monitoring Prometheus/          # Tangkapan layar target UP & query metrik
    ├── 5.bukti monitoring Grafana/             # Tangkapan layar dashboard visualisasi 11 metrik
    ├── 6.bukti alerting Grafana/               # Tangkapan layar 3 alert rules Firing & email notifikasi Gmail
    ├── 7.Inference.py                         # Skrip pengirim trafik simulasi request API
    └── telco_customer_churn_preprocessing.csv # Salinan dataset preprocessed
```

## 🚀 Repositori Terkait

*   **Repositori Eksperimen & Preprocessing**: [https://github.com/mergeconflictlover/Eksperimen_SML_Muhammad-Himbar-Buana](https://github.com/mergeconflictlover/Eksperimen_SML_Muhammad-Himbar-Buana)
*   **Repositori Workflow Retraining CI/CD**: [https://github.com/mergeconflictlover/Workflow-CI_Muhammad-Himbar-Buana](https://github.com/mergeconflictlover/Workflow-CI_Muhammad-Himbar-Buana)
*   **Repositori DagsHub**: [https://dagshub.com/mergeconflictlover/Eksperimen_SML_Muhammad-Himbar-Buana](https://dagshub.com/mergeconflictlover/Eksperimen_SML_Muhammad-Himbar-Buana)
*   **Docker Hub Registry**: [https://hub.docker.com/r/himbarbuana/telco-churn-model](https://hub.docker.com/r/himbarbuana/telco-churn-model)
