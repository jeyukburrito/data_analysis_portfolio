import sys, os
import requests
from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# 스크립트 경로 추가
SCRIPT_DIR = os.path.join(os.getenv('AIRFLOW_HOME', '/opt/airflow'), 'scripts')
sys.path.append(SCRIPT_DIR)
from pipeline import preprocessing_task, train_evaluate_task


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def notify_slack_task(**context: Any) -> None:
    """XCom에서 평가 지표 딕셔너리를 가져와 Slack으로 알림을 보냅니다."""
    
    # 1. XCom에서 결과 딕셔너리 가져오기
    metrics = context['ti'].xcom_pull(task_ids='train_and_evaluate')

    if not isinstance(metrics, dict):
        summary_text = f"모델 평가 결과를 가져오는 데 실패했습니다. XCom 반환값: {metrics}"
    else:
        # 2. 메시지 구성
        summary_text = (
            f"*🚀 모델링 파이프라인 주요 결과 🚀*\n\n"
            f"- *Best Threshold*: `{metrics.get('best_threshold', 0):.4f}`\n"
            f"- *Test ROC AUC*: `{metrics.get('roc_auc', 0):.4f}`\n"
            f"- *Test PR AUC*: `{metrics.get('pr_auc', 0):.4f}`\n\n"
            f"*Classification Report:*"
            f"```{metrics.get('classification_report', 'N/A')}```"
        )

    # 3. Webhook URL을 통해 Slack으로 전송
    webhook_url = Variable.get("slack_url")
    payload = {"text": f":tada: 모델링 파이프라인 완료!\n\n{summary_text}"}
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("Slack 알림 전송 성공")
    except requests.exceptions.RequestException as e:
        print(f"Slack 알림 전송 실패: {e}")
        raise


with DAG(
    dag_id='e2e_modeling_pipeline_with_full_report',
    default_args=default_args,
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'pipeline', 'e2e'],
) as dag:

    # Task 1: 데이터 전처리
    preprocessing = PythonOperator(
        task_id='preprocessing_task',
        python_callable=preprocessing_task,
    )

    # Task 2: 모델 학습 및 평가 (XCom으로 딕셔너리 반환)
    train_and_evaluate = PythonOperator(
        task_id='train_and_evaluate',
        python_callable=train_evaluate_task,
    )

    # Task 3: Slack 알림
    send_notification = PythonOperator(
        task_id='send_slack_notification',
        python_callable=notify_slack_task,
        provide_context=True
    )

    preprocessing >> train_and_evaluate >> send_notification
