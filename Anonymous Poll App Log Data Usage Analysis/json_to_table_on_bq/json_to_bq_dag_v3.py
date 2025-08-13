from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from plugins.process_json_to_table import process_monthly_json_to_bq

# 날짜별 업로드할 월 매핑
UPLOAD_SCHEDULE = {
    "2025-08-06": "2023-08",
    "2025-08-07": "2023-09",
    "2025-08-08": "2023-10",
    "2025-08-09": "2023-11",
    "2025-08-10": "2023-12",
    "2025-08-11": "2024-01"
}

# GCS / BQ 설정
BUCKET_NAME = "json_to_table"
BQ_DATASET = "json_to_table"
BQ_PROJECT = "codeit-final-project"

def upload_for_date(execution_date_str, **context):
    year_month = UPLOAD_SCHEDULE.get(execution_date_str)
    if not year_month:
        raise ValueError(f"No upload schedule for {execution_date_str}")

    print(f"Uploading data for folder: {year_month}")
    process_monthly_json_to_bq(
        bucket_name=BUCKET_NAME,
        folder_path=f"{year_month}/",  # 예: "2023-10/"
        bq_dataset=BQ_DATASET,
        bq_project=BQ_PROJECT
    )

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="json_to_bq_dag_v4",
    default_args=default_args,
    description="Upload specific month data based on date",
    schedule_interval="@daily",
    start_date=datetime(2025, 8, 6),
    catchup=False,
) as dag:

    upload_task = PythonOperator(
        task_id="upload_task",
        python_callable=upload_for_date,
        op_args=["{{ ds }}"],  # execution_date를 YYYY-MM-DD 문자열로 전달
        provide_context=True
    )

    upload_task