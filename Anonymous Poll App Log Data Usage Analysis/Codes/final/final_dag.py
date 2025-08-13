from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

# 실행 날짜(YYYY-MM-DD)와 예측에 사용할 빅쿼리 테이블 접미사(YYYYMM)를 매핑합니다.
UPLOAD_SCHEDULE = {
    "2025-08-09": "202308",
    "2025-08-10": "202309",
    "2025-08-11": "202310",
    "2025-08-12": "202311",
    "2025-08-13": "202312",
    "2025-08-14": "202401",
}

# --------------------------------------------------------------------------------
# BigQuery ML 쿼리 정의
# --------------------------------------------------------------------------------

# 1단계: 모델 훈련 쿼리
TRAIN_MODEL_SQL = """
CREATE OR REPLACE MODEL `codeit-final-project.json_to_table.payment_prediction_model_{{ get_suffix(ds) }}`
OPTIONS(
  MODEL_TYPE='BOOSTED_TREE_CLASSIFIER',
  INPUT_LABEL_COLS=['actual_payment']
) AS
WITH engineered_features AS (
    WITH user_events AS (
        SELECT t1.user_id, t2.event_datetime, t2.event_key, t1.session_id, t1.osname
        FROM `codeit-final-project.json_to_table.hackle_properties_{{ get_suffix(ds) }}` AS t1
        JOIN `codeit-final-project.json_to_table.hackle_events_{{ get_suffix(ds) }}` AS t2 ON t1.session_id = t2.session_id
        WHERE t1.user_id IS NOT NULL AND t1.user_id != 'nan'
    ), user_stats AS (
        SELECT user_id, COUNT(DISTINCT session_id) AS total_sessions, COUNT(DISTINCT DATE(event_datetime)) AS active_days, COUNT(event_key) AS total_events, SUM(CASE WHEN event_key = 'complete_purchase' THEN 1 ELSE 0 END) AS core_events, MAX(osname) AS osname FROM user_events GROUP BY user_id
    ), user_features AS (
        SELECT t1.*, t2.gender, t2.is_push_on, t2.ban_status, t2.report_count, t2.pending_chat, t2.pending_votes,
               -- 모든 ARRAY_LENGTH 함수에 JSON_EXTRACT_ARRAY 및 COALESCE 적용
               COALESCE(ARRAY_LENGTH(JSON_EXTRACT_ARRAY(t2.friend_id_list)), 0) AS friend_id_count,
               COALESCE(ARRAY_LENGTH(JSON_EXTRACT_ARRAY(t2.block_user_id_list)), 0) AS block_user_id_count,
               COALESCE(ARRAY_LENGTH(JSON_EXTRACT_ARRAY(t2.hide_user_id_list)), 0) AS hide_user_id_count,
               t3.school_type, t3.address, t4.grade, t4.user_id AS user_props_id, t4.class, t5.payment_status, 0 as cluster
        FROM user_stats AS t1
        LEFT JOIN `codeit-final-project.json_to_table.accounts_user` AS t2 ON t1.user_id = CAST(t2.id AS STRING)
        LEFT JOIN `codeit-final-project.json_to_table.accounts_school` AS t3 ON t2.group_id = t3.id
        LEFT JOIN `codeit-final-project.json_to_table.user_properties_{{ get_suffix(ds) }}` AS t4 ON t1.user_id = t4.user_id
        LEFT JOIN (SELECT user_id, 1 as payment_status FROM `codeit-final-project.json_to_table.accounts_paymenthistory` GROUP BY user_id) AS t5 ON t1.user_id = CAST(t5.user_id AS STRING)
    )
    SELECT *, COALESCE(payment_status, 0) AS actual_payment, CONCAT(school_type, CAST(grade AS STRING)) AS school_grade,
           CASE WHEN starts_with(address, '강원도') THEN 'Gangwon' WHEN starts_with(address, '경기도') THEN 'Gyeonggi' WHEN starts_with(address, '충청남도') THEN 'Chungnam' WHEN starts_with(address, '충청북도') THEN 'Chungbuk' WHEN starts_with(address, '경상북도') THEN 'Gyeongbuk' WHEN starts_with(address, '경상남도') THEN 'Gyeongnam' WHEN starts_with(address, '광주광역시') THEN 'Gwangju' WHEN starts_with(address, '대구광역시') THEN 'Daegu' WHEN starts_with(address, '대전광역시') THEN 'Daejeon' WHEN starts_with(address, '부산광역시') THEN 'Busan' WHEN starts_with(address, '울산광역시') THEN 'Ulsan' WHEN starts_with(address, '인천광역시') THEN 'Incheon' WHEN starts_with(address, '서울특별시') THEN 'Seoul' WHEN starts_with(address, '세종특별자치시') THEN 'Sejong' WHEN starts_with(address, '제주특별자치도') THEN 'Jeju' WHEN starts_with(address, '전라남도') THEN 'Jeonnam' WHEN starts_with(address, '전라북도') THEN 'Jeonbuk' ELSE 'Unknown' END AS province
    FROM user_features
)
SELECT * FROM engineered_features;
"""

# 2단계: 모델 성능 평가 쿼리
EVALUATE_MODEL_SQL = """
CREATE OR REPLACE TABLE `codeit-final-project.json_to_table.payment_evaluation_results_{{ get_suffix(ds) }}` AS
SELECT *
FROM ML.EVALUATE(MODEL `codeit-final-project.json_to_table.payment_prediction_model_{{ get_suffix(ds) }}`);
"""

# 3단계: 예측 결과 저장 쿼리
PREDICT_SQL = """
CREATE OR REPLACE TABLE `codeit-final-project.json_to_table.payment_prediction_results_{{ get_suffix(ds) }}` AS
SELECT *
FROM ML.PREDICT(MODEL `codeit-final-project.json_to_table.payment_prediction_model_{{ get_suffix(ds) }}`,
  (
    -- 훈련 쿼리와 동일하게 수정
    WITH engineered_features AS (
        WITH user_events AS (
            SELECT t1.user_id, t2.event_datetime, t2.event_key, t1.session_id, t1.osname
            FROM `codeit-final-project.json_to_table.hackle_properties_{{ get_suffix(ds) }}` AS t1
            JOIN `codeit-final-project.json_to_table.hackle_events_{{ get_suffix(ds) }}` AS t2 ON t1.session_id = t2.session_id
            WHERE t1.user_id IS NOT NULL AND t1.user_id != 'nan'
        ), user_stats AS (
            SELECT user_id, COUNT(DISTINCT session_id) AS total_sessions, COUNT(DISTINCT DATE(event_datetime)) AS active_days, COUNT(event_key) AS total_events, SUM(CASE WHEN event_key = 'complete_purchase' THEN 1 ELSE 0 END) AS core_events, MAX(osname) AS osname FROM user_events GROUP BY user_id
        ), user_features AS (
            SELECT t1.*, t2.gender, t2.is_push_on, t2.ban_status, t2.report_count, t2.pending_chat, t2.pending_votes,
                   -- 모든 ARRAY_LENGTH 함수에 JSON_EXTRACT_ARRAY 및 COALESCE 적용
                   COALESCE(ARRAY_LENGTH(JSON_EXTRACT_ARRAY(t2.friend_id_list)), 0) AS friend_id_count,
                   COALESCE(ARRAY_LENGTH(JSON_EXTRACT_ARRAY(t2.block_user_id_list)), 0) AS block_user_id_count,
                   COALESCE(ARRAY_LENGTH(JSON_EXTRACT_ARRAY(t2.hide_user_id_list)), 0) AS hide_user_id_count,
                   t3.school_type, t3.address, t4.grade, t4.user_id AS user_props_id, t4.class, t5.payment_status, 0 as cluster
            FROM user_stats AS t1
            LEFT JOIN `codeit-final-project.json_to_table.accounts_user` AS t2 ON t1.user_id = CAST(t2.id AS STRING)
            LEFT JOIN `codeit-final-project.json_to_table.accounts_school` AS t3 ON t2.group_id = t3.id
            LEFT JOIN `codeit-final-project.json_to_table.user_properties_{{ get_suffix(ds) }}` AS t4 ON t1.user_id = t4.user_id
            LEFT JOIN (SELECT user_id, 1 as payment_status FROM `codeit-final-project.json_to_table.accounts_paymenthistory` GROUP BY user_id) AS t5 ON t1.user_id = CAST(t5.user_id AS STRING)
        )
        SELECT *, COALESCE(payment_status, 0) AS actual_payment, CONCAT(school_type, CAST(grade AS STRING)) AS school_grade,
               CASE WHEN starts_with(address, '강원도') THEN 'Gangwon' WHEN starts_with(address, '경기도') THEN 'Gyeonggi' WHEN starts_with(address, '충청남도') THEN 'Chungnam' WHEN starts_with(address, '충청북도') THEN 'Chungbuk' WHEN starts_with(address, '경상북도') THEN 'Gyeongbuk' WHEN starts_with(address, '경상남도') THEN 'Gyeongnam' WHEN starts_with(address, '광주광역시') THEN 'Gwangju' WHEN starts_with(address, '대구광역시') THEN 'Daegu' WHEN starts_with(address, '대전광역시') THEN 'Daejeon' WHEN starts_with(address, '부산광역시') THEN 'Busan' WHEN starts_with(address, '울산광역시') THEN 'Ulsan' WHEN starts_with(address, '인천광역시') THEN 'Incheon' WHEN starts_with(address, '서울특별시') THEN 'Seoul' WHEN starts_with(address, '세종특별자치시') THEN 'Sejong' WHEN starts_with(address, '제주특별자치도') THEN 'Jeju' WHEN starts_with(address, '전라남도') THEN 'Jeonnam' WHEN starts_with(address, '전라북도') THEN 'Jeonbuk' ELSE 'Unknown' END AS province
        FROM user_features
    )
    SELECT * FROM engineered_features
  )
);
"""

# --------------------------------------------------------------------------------
# Airflow DAG 정의
# --------------------------------------------------------------------------------
with DAG(
    dag_id="daily_payment_prediction_pipeline",
    start_date=pendulum.datetime(2025, 8, 9, tz="Asia/Seoul"),
    end_date=pendulum.datetime(2025, 8, 14, tz="Asia/Seoul"),
    schedule="@daily",
    catchup=True,
    tags=["gcp", "bigquery", "bqml"],
    user_defined_macros={
        "get_suffix": lambda ds: UPLOAD_SCHEDULE.get(ds),
    },
) as dag:
    train_model = BigQueryInsertJobOperator(
        task_id="train_model",
        configuration={
            "query": {
                "query": TRAIN_MODEL_SQL,
                "useLegacySql": False,
            }
        },
        gcp_conn_id="google_cloud_default",
    )

    evaluate_model = BigQueryInsertJobOperator(
        task_id="evaluate_model",
        configuration={
            "query": {
                "query": EVALUATE_MODEL_SQL,
                "useLegacySql": False,
            }
        },
        gcp_conn_id="google_cloud_default",
    )

    predict_and_save = BigQueryInsertJobOperator(
        task_id="predict_and_save",
        configuration={
            "query": {
                "query": PREDICT_SQL,
                "useLegacySql": False,
            }
        },
        gcp_conn_id="google_cloud_default",
    )

    train_model >> evaluate_model >> predict_and_save