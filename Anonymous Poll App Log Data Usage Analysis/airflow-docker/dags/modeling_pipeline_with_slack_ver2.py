import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# --- 상수 정의 ---
# Airflow 환경 변수 및 기본 경로 설정
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
SCRIPT_DIR = os.path.join(AIRFLOW_HOME, 'scripts')
RESULTS_ROOT = os.path.join(AIRFLOW_HOME, 'results')

# Slack 메시지 파싱에 사용될 키
METRIC_KEYS = ["Best Threshold", "Test ROC AUC", "Test PR AUC"]

# 스크립트 경로 추가
sys.path.append(SCRIPT_DIR)
from pipeline import preprocessing_task, train_evaluate_task


# --- Helper 함수들 ---

def _get_results_path(execution_date_str: str) -> str:
    """Airflow 실행 날짜를 기반으로 결과 디렉토리 경로를 반환합니다."""
    dt = datetime.strptime(execution_date_str, '%Y-%m-%d')
    return os.path.join(RESULTS_ROOT, f"{dt:%Y%m%d}_baseline_lgbm_results")

def _parse_summary_file(summary_path: str) -> Dict[str, str]:
    """결과 요약 파일을 읽고 주요 지표를 파싱하여 딕셔너리로 반환합니다."""
    metrics = {}
    with open(summary_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(':', 1)
                key = key.strip()
                if key in METRIC_KEYS:
                    metrics[key] = value.strip()
    return metrics

def _create_slack_message(metrics: Dict[str, str], results_dir: str) -> str:
    """파싱된 지표를 사용하여 Slack 메시지 본문을 생성합니다."""
    if not metrics:
        return f"주요 지표를 파싱할 수 없습니다. 결과 폴더를 확인해주세요: `{results_dir}`"

    return (
        f"*🚀 모델링 파이프라인 주요 결과 🚀*\n\n"
        f"- *Best Threshold*: `{metrics.get('Best Threshold', 'N/A')}`\n"
        f"- *Test ROC AUC*: `{metrics.get('Test ROC AUC', 'N/A')}`\n"
        f"- *Test PR AUC*: `{metrics.get('Test PR AUC', 'N/A')}`\n\n"
        f"*상세 정보*\n"
        f"- 전체 리포트, SHAP 시각화, PR Curve 이미지는 아래 경로에 저장되었습니다.\n"
        f"- 경로: `{results_dir}`"
    )


# --- Airflow Task 함수 ---

def notify_slack_task(**context: Any) -> None:
    """결과를 취합하여 Slack으로 알림을 보내는 Airflow Task입니다."""
    execution_date = context['ds']
    results_dir = _get_results_path(execution_date)
    summary_path = os.path.join(results_dir, 'model_evaluation_summary.txt')

    try:
        metrics = _parse_summary_file(summary_path)
        summary_text = _create_slack_message(metrics, results_dir)
    except FileNotFoundError:
        summary_text = f"결과 요약 파일을 찾을 수 없습니다: {summary_path}"
    except Exception as e:
        summary_text = f"결과 처리 중 오류 발생: {e}"

    webhook_url = Variable.get("slack_url")
    payload = {
        "text": f":tada: 모델링 파이프라인이 성공적으로 완료되었습니다!\n\n{summary_text}"
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print("Slack 알림 전송 성공")
    except requests.exceptions.RequestException as e:
        print(f"Slack 알림 전송 실패: {e}")
        raise


# --- DAG 정의 ---

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'provide_context': True,  # 모든 Operator에 context 제공
}

with DAG(
    dag_id='modeling_pipeline_with_slack_v3',
    default_args=default_args,
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'pipeline', 'refactored'],
) as dag:

    preprocess = PythonOperator(
        task_id='preprocessing_task',
        python_callable=preprocessing_task,
    )

    train_evaluate = PythonOperator(
        task_id='train_evaluate_task',
        python_callable=train_evaluate_task,
    )

    send_slack = PythonOperator(
        task_id='send_slack_notification',
        python_callable=notify_slack_task,
    )

    preprocess >> train_evaluate >> send_slack
