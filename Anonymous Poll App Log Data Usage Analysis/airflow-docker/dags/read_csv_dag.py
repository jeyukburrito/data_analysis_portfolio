# dags/read_user_data.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_and_log():
    # 컨테이너 안 경로로 파일 읽기
    path = '/opt/airflow/data/filtered_user_with_school_id.csv'
    df = pd.read_csv(path)
    # 간단히 첫 다섯 줄 로그로 출력
    print(df.head())

with DAG(
    dag_id='read_user_data',
    default_args=default_args,
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,   # 수동 실행용
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='load_and_log',
        python_callable=load_and_log
    )
