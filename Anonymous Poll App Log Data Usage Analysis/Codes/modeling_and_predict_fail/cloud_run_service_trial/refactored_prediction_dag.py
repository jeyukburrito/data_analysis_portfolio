from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from datetime import datetime

# 실행 날짜와 테이블 접미사 매핑
UPLOAD_SCHEDULE = {
    "2025-08-06": "202308", "2025-08-07": "202309", "2025-08-08": "202310",
    "2025-08-09": "202311", "2025-08-10": "202312", "2025-08-11": "202401"
}

TABLE_SUFFIX = UPLOAD_SCHEDULE.get("{{ ds }}", "202308") # ds는 Airflow의 실행 날짜 템플릿 변수

# BigQuery SQL 쿼리 정의
# 원본 쿼리를 기반으로, 최종 SELECT 구문에 원격 함수 호출을 추가
PREDICTION_QUERY = f"""
CREATE OR REPLACE TABLE `codeit-final-project.json_to_table.payment_prediction_result_{TABLE_SUFFIX}` AS
WITH user_events AS (
    SELECT
        t1.user_id, t2.event_datetime, t2.event_key, t1.session_id, t1.osname
    FROM `codeit-final-project.json_to_table.hackle_properties_{TABLE_SUFFIX}` AS t1
    JOIN `codeit-final-project.json_to_table.hackle_events_{TABLE_SUFFIX}` AS t2 ON t1.session_id = t2.session_id
    WHERE t1.user_id IS NOT NULL AND t1.user_id != 'nan'
),
user_stats AS (
    SELECT
        user_id,
        COUNT(DISTINCT session_id) AS total_sessions,
        COUNT(DISTINCT DATE(event_datetime)) AS active_days,
        COUNT(event_key) AS total_events,
        SUM(CASE WHEN event_key = 'complete_purchase' THEN 1 ELSE 0 END) AS core_events,
        MAX(osname) AS osname
    FROM user_events
    GROUP BY user_id
),
user_features AS (
    SELECT
        t1.*, t2.gender, t2.is_push_on, t2.ban_status, t2.report_count,
        t2.pending_chat, t2.pending_votes, ARRAY_LENGTH(t2.friend_id_list) AS friend_id_count,
        ARRAY_LENGTH(t2.block_user_id_list) AS block_user_id_count,
        ARRAY_LENGTH(t2.hide_user_id_list) AS hide_user_id_count, t3.school_type, t3.address,
        t4.grade, t4.id AS user_props_id, t4.class,
        t5.payment_status,
        0 as cluster -- TODO: 클러스터링 로직 반영 필요
    FROM user_stats AS t1
    LEFT JOIN `codeit-final-project.json_to_table.accounts_user` AS t2 ON t1.user_id = CAST(t2.id AS STRING)
    LEFT JOIN `codeit-final-project.json_to_table.accounts_school` AS t3 ON t2.group_id = t3.id
    LEFT JOIN `codeit-final-project.json_to_table.user_properties_{TABLE_SUFFIX}` AS t4 ON t1.user_id = t4.id
    LEFT JOIN (SELECT user_id, 1 as payment_status FROM `codeit-final-project.json_to_table.accounts_paymenthistory` GROUP BY user_id) AS t5 ON t1.user_id = CAST(t5.user_id AS STRING)
),
engineered_features AS (
    SELECT
        *,
        COALESCE(payment_status, 0) AS actual_payment, -- NULL을 0으로 처리
        CONCAT(school_type, CAST(grade AS STRING)) AS school_grade,
        CASE
            WHEN starts_with(address, '강원도') THEN 'Gangwon' WHEN starts_with(address, '경기도') THEN 'Gyeonggi'
            WHEN starts_with(address, '충청남도') THEN 'Chungnam' WHEN starts_with(address, '충청북도') THEN 'Chungbuk'
            WHEN starts_with(address, '경상북도') THEN 'Gyeongbuk' WHEN starts_with(address, '경상남도') THEN 'Gyeongnam'
            WHEN starts_with(address, '광주광역시') THEN 'Gwangju' WHEN starts_with(address, '대구광역시') THEN 'Daegu'
            WHEN starts_with(address, '대전광역시') THEN 'Daejeon' WHEN starts_with(address, '부산광역시') THEN 'Busan'
            WHEN starts_with(address, '울산광역시') THEN 'Ulsan' WHEN starts_with(address, '인천광역시') THEN 'Incheon'
            WHEN starts_with(address, '서울특별시') THEN 'Seoul' WHEN starts_with(address, '세종특별자치시') THEN 'Sejong'
            WHEN starts_with(address, '제주특별자치도') THEN 'Jeju' WHEN starts_with(address, '전라남도') THEN 'Jeonnam'
            WHEN starts_with(address, '전라북도') THEN 'Jeonbuk' ELSE 'Unknown'
        END AS province
    FROM user_features
)
SELECT
  user_id,
  actual_payment,
  -- 원격 함수 호출로 예측 수행
  `json_to_table.predict_payment`(
      total_events, total_sessions, active_days, core_events, school_grade, osname, friend_id_count,
      gender, 0, report_count, pending_votes, ban_status, is_push_on, class, hide_user_id_count,
      block_user_id_count, province, cluster, pending_chat, user_props_id
  ) as predicted_payment
FROM engineered_features
"""

with DAG(
    dag_id='payment_prediction_pipeline_v5_refactored',
    start_date=datetime(2025, 8, 6),
    schedule_interval='0 3 * * *',
    catchup=True, # 스케줄에 맞춰 과거 작업도 실행
    tags=['gcp', 'bigquery', 'prediction', 'stable'],
) as dag:

    # 태스크 1: BigQuery에서 모든 전처리, 예측을 수행하고 결과를 새 테이블에 저장
    run_prediction_in_bigquery = BigQueryExecuteQueryOperator(
        task_id='run_prediction_in_bigquery',
        sql=PREDICTION_QUERY,
        use_legacy_sql=False,
        gcp_conn_id='google_cloud_default', # Airflow Connection ID
    )

    # 태스크 2: (선택 사항) 예측 결과를 GCS에 CSV 파일로 저장
    export_results_to_gcs = BigQueryToGCSOperator(
        task_id='export_results_to_gcs',
        source_project_dataset_table=f"codeit-final-project.json_to_table.payment_prediction_result_{TABLE_SUFFIX}",
        destination_cloud_storage_uris=[f"gs://preprocess_and_predict_results/payment_prediction_result_{TABLE_SUFFIX}.csv"],
        export_format='CSV',
        gcp_conn_id='google_cloud_default',
    )

    run_prediction_in_bigquery >> export_results_to_gcs