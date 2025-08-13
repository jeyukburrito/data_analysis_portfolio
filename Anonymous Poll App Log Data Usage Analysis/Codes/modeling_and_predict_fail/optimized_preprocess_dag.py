from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from optimized_pipeline import main

# 실행 날짜(YYYY-MM-DD)와 빅쿼리 테이블 접미사(YYYYMM) 매핑
UPLOAD_SCHEDULE = {
    "2025-08-06": "202308",
    "2025-08-07": "202309",
    "2025-08-08": "202310",
    "2025-08-09": "202311",
    "2025-08-10": "202312",
    "2025-08-11": "202401"
}

def get_table_suffix(ds, **kwargs):
    """실행 날짜에 맞는 테이블 접미사를 반환합니다."""
    table_suffix = UPLOAD_SCHEDULE.get(ds)
    if not table_suffix:
        raise ValueError(f"No table suffix found for execution date: {ds}")
    return table_suffix

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 8, 6),
    'retries': 1,
}

with DAG(
    dag_id='payment_prediction_pipeline_v4',
    default_args=default_args,
    schedule_interval='0 3 * * *',
    tags=['gcp', 'bigquery', 'prediction'],
    catchup=False,
) as dag:
    
    get_suffix_task = PythonOperator(
        task_id='get_table_suffix',
        python_callable=get_table_suffix,
        provide_context=True,
    )

    run_prediction_pipeline = PythonOperator(
        task_id='preprocess_and_predict',
        python_callable=main,
        op_kwargs={
            'table_suffix': "{{ task_instance.xcom_pull(task_ids='get_table_suffix') }}",
            'output_bucket': 'preprocess_and_predict_results',
            'project_id': 'codeit-final-project',
            'dataset_id': 'json_to_table',
        },
    )

    get_suffix_task >> run_prediction_pipeline