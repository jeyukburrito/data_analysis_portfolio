# Use the official Airflow image as base
FROM apache/airflow:2.9.1

# Switch to root to install system dependencies
USER root

# Install OpenMP runtime needed by LightGBM
RUN apt-get update \
    && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user to install Python packages
USER airflow

# Install Python dependencies for ML pipeline as airflow user
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    imbalanced-learn \
    lightgbm \
    statsmodels \
    apache-airflow-providers-slack \
    slack-sdk \
    shap
