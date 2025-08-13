# dags/modeling_pipeline_with_slack.py

import sys, os
import requests
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# 스크립트 경로 추가
sys.path.append(os.path.join(os.getenv('AIRFLOW_HOME','/opt/airflow'), 'scripts'))
from pipeline import run_full_pipeline  # 문자열을 return 합니다

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def notify_slack(**context):
    # 1) 파이프라인 결과 요약 가져오기
    summary = context['ti'].xcom_pull(task_ids='run_full_pipeline')
    # 2) Webhook URL 변수에서 불러오기
    webhook_url = Variable.get("slack_url")
    # 3) 페이로드 구성
    payload = {
        "text": f":tada: 모델링 파이프라인 완료!\n```{summary}```"
    }
    # 4) POST 요청
    resp = requests.post(webhook_url, json=payload)
    resp.raise_for_status()

with DAG(
    dag_id='modeling_pipeline_with_slack',
    default_args=default_args,
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    # 1) 전체 파이프라인 실행
    run_full = PythonOperator(
        task_id='run_full_pipeline',
        python_callable=run_full_pipeline
    )

    # 2) Slack 알림 (PythonOperator로 직접 구현)
    send_slack = PythonOperator(
        task_id='send_slack',
        python_callable=notify_slack,
        provide_context=True
    )

    run_full >> send_slack
