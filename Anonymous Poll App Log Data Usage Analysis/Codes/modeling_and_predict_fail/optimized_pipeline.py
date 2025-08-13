import pandas as pd
import numpy as np
import os
import pickle
from google.cloud import bigquery, storage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# --- 1. 환경 변수 설정 (이제 DAG에서 값을 받으므로, 이 값들은 기본값 역할만 합니다.) ---
TABLE_PREFIX = 'hackle'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'payment_prediction_model.pkl')


# --- 2. BigQuery 데이터 로드 함수 (SQL 기반으로 모든 테이블 조인) ---
def load_and_aggregate_from_bigquery(project_id, dataset_id, table_suffix):
    """
    BigQuery SQL 쿼리를 사용하여 데이터를 조인하고 집계한 후,
    최종 결과(user_summary)를 Pandas DataFrame으로 반환합니다.
    """
    print(f"--- 2. BigQuery에서 데이터 집계 시작 (테이블: {table_suffix}) ---")
    
    client = bigquery.Client(project=project_id)
    
    # 테이블 이름 정의
    events_table = f'`{project_id}.{dataset_id}.{TABLE_PREFIX}_events_{table_suffix}`'
    properties_table = f'`{project_id}.{dataset_id}.{TABLE_PREFIX}_properties_{table_suffix}`'
    user_table = f'`{project_id}.{dataset_id}.accounts_user`'
    school_table = f'`{project_id}.{dataset_id}.accounts_school`'
    payment_history_table = f'`{project_id}.{dataset_id}.accounts_paymenthistory`'
    user_properties_table = f'`{project_id}.{dataset_id}.user_properties_{table_suffix}`'
    
    # SQL 쿼리 작성 - 모든 전처리/피처 엔지니어링을 빅쿼리에서 수행
    query = f"""
    WITH user_events AS (
        SELECT
            t1.user_id,
            t2.event_datetime,
            t2.event_key,
            t1.session_id,
            t1.osname
        FROM
            {properties_table} AS t1
        JOIN
            {events_table} AS t2
            ON t1.session_id = t2.session_id
        WHERE
            t1.user_id IS NOT NULL AND t1.user_id != 'nan'
    ),
    user_stats AS (
        SELECT
            user_id,
            COUNT(DISTINCT session_id) AS total_sessions,
            COUNT(DISTINCT DATE(event_datetime)) AS active_days,
            COUNT(event_key) AS total_events,
            SUM(CASE WHEN event_key = 'complete_purchase' THEN 1 ELSE 0 END) AS core_events,
            MAX(osname) AS osname
        FROM
            user_events
        GROUP BY
            user_id
    ),
    user_features AS (
        SELECT
            t1.*,
            t2.gender,
            t2.is_push_on,
            t2.ban_status,
            t2.report_count,
            t2.pending_chat,
            t2.pending_votes,
            ARRAY_LENGTH(t2.friend_id_list) AS friend_id_count,
            ARRAY_LENGTH(t2.block_user_id_list) AS block_user_id_count,
            ARRAY_LENGTH(t2.hide_user_id_list) AS hide_user_id_count,
            t3.school_type,
            t3.address,
            t4.grade,
            t4.id AS user_props_id,
            t4.class,
            t5.payment_status,
            t6.cluster
        FROM
            user_stats AS t1
        LEFT JOIN
            {user_table} AS t2
            ON t1.user_id = CAST(t2.id AS STRING)
        LEFT JOIN
            {school_table} AS t3
            ON t2.group_id = t3.id
        LEFT JOIN
            {user_properties_table} AS t4
            ON t1.user_id = t4.id
        LEFT JOIN
            (SELECT user_id, 1 as payment_status FROM {payment_history_table} GROUP BY user_id) AS t5
            ON t1.user_id = CAST(t5.user_id AS STRING)
        LEFT JOIN
            # TODO: 클러스터링 결과 테이블을 조인해야 합니다.
            # 이 쿼리에서는 임시로 0으로 설정합니다.
            (SELECT user_id, 0 as cluster FROM {properties_table}) AS t6
            ON t1.user_id = t6.user_id
    )
    SELECT
        user_id,
        total_sessions,
        active_days,
        total_events,
        core_events,
        payment_status,
        -- 추가 피처 엔지니어링
        CONCAT(school_type, CAST(grade AS STRING)) AS school_grade,
        CASE
            WHEN starts_with(address, '강원도') THEN 'Gangwon'
            WHEN starts_with(address, '경기도') THEN 'Gyeonggi'
            WHEN starts_with(address, '충청남도') THEN 'Chungnam'
            WHEN starts_with(address, '충청북도') THEN 'Chungbuk'
            WHEN starts_with(address, '경상북도') THEN 'Gyeongbuk'
            WHEN starts_with(address, '경상남도') THEN 'Gyeongnam'
            WHEN starts_with(address, '광주광역시') THEN 'Gwangju'
            WHEN starts_with(address, '대구광역시') THEN 'Daegu'
            WHEN starts_with(address, '대전광역시') THEN 'Daejeon'
            WHEN starts_with(address, '부산광역시') THEN 'Busan'
            WHEN starts_with(address, '울산광역시') THEN 'Ulsan'
            WHEN starts_with(address, '인천광역시') THEN 'Incheon'
            WHEN starts_with(address, '서울특별시') THEN 'Seoul'
            WHEN starts_with(address, '세종특별자치시') THEN 'Sejong'
            WHEN starts_with(address, '제주특별자치도') THEN 'Jeju'
            WHEN starts_with(address, '전라남도') THEN 'Jeonnam'
            WHEN starts_with(address, '전라북도') THEN 'Jeonbuk'
            ELSE NULL
        END AS province,
        -- 모델에 필요한 나머지 피처들
        gender,
        is_push_on,
        ban_status,
        report_count,
        pending_chat,
        pending_votes,
        friend_id_count,
        block_user_id_count,
        hide_user_id_count,
        user_props_id,
        class,
        cluster,
        osname
    FROM
        user_features
    """
    
    print("BigQuery 쿼리 실행...")
    df = client.query(query).to_dataframe()
    
    print(f"BigQuery에서 집계된 데이터 크기: {df.shape}")
    return df

# --- 3. 사용자 요약 데이터 생성 (기존 user_summary_script.py 내용) ---
def create_user_summary(df, table_suffix):
    """
    BigQuery에서 가져온 요약 데이터에 추가 피처를 엔지니어링합니다.
    """
    print("--- 3. 사용자 요약 데이터에 추가 피처 엔지니어링 ---")
    
    df = df.fillna(0)
    df['payment_status'] = df['payment_status'].astype(bool)

    print(f"사용자 요약 데이터 생성 완료. 총 사용자 수: {len(df)}")
    return df

# --- 4. 모델 로드, 예측 및 성능 평가 ---
def predict_and_evaluate(user_summary_df):
    """pkl 모델을 로드하여 결제 여부를 예측하고 성능을 평가합니다."""
    print("--- 4. 모델 예측 및 성능 평가 시작 ---")
    
    features = ['total_events', 'total_sessions', 'active_days', 'core_events', 'school_grade', 'osname', 'friend_id_count', 'gender', 'alarm_count', 'report_count', 'pending_votes', 'ban_status', 'is_push_on', 'class', 'hide_user_id_count', 'block_user_id_count', 'province', 'cluster', 'pending_chat', 'id']
    X = user_summary_df[features]
    y_true = user_summary_df['payment_status']

    if not os.path.exists(MODEL_PATH):
        print(f"경고: 모델 파일이 '{MODEL_PATH}'에 존재하지 않습니다. 예측 단계를 건너뜁니다.")
        return user_summary_df

    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
        
    print(f"모델 '{MODEL_PATH}' 로드 완료.")
    
    y_prob = pipeline.predict_proba(X)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    best_thresh_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_thresh_idx]
    y_pred_optim = (y_prob >= best_thresh).astype(int)

    accuracy = accuracy_score(y_true, y_pred_optim)
    precision = precision_score(y_true, y_pred_optim)
    recall = recall_score(y_true, y_pred_optim)
    f1 = f1_score(y_true, y_pred_optim)

    print("\n--- 예측 성능 평가 결과 ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    user_summary_df['predicted_payment'] = y_pred_optim

    return user_summary_df

# --- Main 실행 함수 ---
def main(table_suffix, output_bucket, project_id, dataset_id):
    """전체 파이프라인을 실행하고 결과를 GCS 버킷에 저장합니다."""
    
    print(f"--- 파이프라인 실행 시작 (테이블: {table_suffix}, 버킷: {output_bucket}) ---")
    
    storage_client = storage.Client(project=project_id)
    
    df_summary = load_and_aggregate_from_bigquery(project_id, dataset_id, table_suffix)
    df_summary_with_features = create_user_summary(df_summary, table_suffix)
    df_result = predict_and_evaluate(df_summary_with_features)
    
    output_filename = f'payment_prediction_result_{table_suffix}.csv'
    
    try:
        bucket = storage_client.bucket(output_bucket)
        blob = bucket.blob(output_filename)
        
        df_result.to_csv(f'/tmp/{output_filename}', index=False, encoding='utf-8-sig')
        blob.upload_from_filename(f'/tmp/{output_filename}')
        
        print(f"최종 결과 데이터가 'gs://{output_bucket}/{output_filename}'에 성공적으로 저장되었습니다.")
        os.remove(f'/tmp/{output_filename}')
        
    except Exception as e:
        print(f"오류: GCS 업로드 실패 - {e}")
        raise
        
if __name__ == '__main__':
    main('202308', 'your-gcs-bucket-name', 'your-gcp-project-id', 'your-dataset-id')