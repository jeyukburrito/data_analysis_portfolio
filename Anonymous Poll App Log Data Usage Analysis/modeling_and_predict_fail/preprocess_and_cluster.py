import pandas as pd
import numpy as np
import re
import uuid
import os
import pickle
from google.cloud import bigquery, storage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# --- 1. 환경 변수 설정 ---
PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-gcp-project-id')
DATASET_ID = 'your-dataset-id'
TABLE_PREFIX = 'hackle'
MODEL_PATH = '/opt/airflow/dags/models/payment_prediction_model.pkl'

# BigQuery Client 초기화
client = bigquery.Client(project=PROJECT_ID)
# GCS Client 초기화
storage_client = storage.Client(project=PROJECT_ID)

# --- 2. BigQuery 데이터 로드 함수 ---
def load_data_from_bigquery(table_name):
    """지정된 테이블에서 데이터를 읽어와 Pandas DataFrame으로 반환합니다."""
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`"
    print(f"Loading data from BigQuery table: {table_name}")
    df = client.query(query).to_dataframe()
    return df

# --- 3. 이벤트 데이터 전처리 (event_processing_script.py 통합) ---
def preprocess_events(df_events, df_properties):
    """이벤트 데이터와 프로퍼티 데이터를 병합하고, 세션을 재정의하여 중복을 제거합니다."""
    print("--- 3. 이벤트 데이터 전처리 시작 ---")

    # 데이터 타입 변환
    df_properties['session_id'] = df_properties['session_id'].astype(str)
    df_properties['device_id'] = df_properties['device_id'].astype(str)
    df_properties['user_id'] = df_properties['user_id'].astype(str)
    
    # session_id별 user_id, device_id 매핑
    session_map = df_properties.groupby('session_id')[['user_id', 'device_id']].first().reset_index()
    df = df_events.merge(session_map, on='session_id', how='left')
    df['event_datetime'] = pd.to_datetime(df['event_datetime'], errors='coerce')
    df_clean_initial = df[df['user_id'] != 'nan'].copy()

    # ID 규칙 검증 함수
    def is_numeric_user_id(user_id_str):
        return bool(re.fullmatch(r'\d+', user_id_str))
    def is_valid_session_id(session_id_str):
        return bool(re.search(r'[a-zA-Z]', session_id_str)) and '-' not in session_id_str
    def is_valid_device_id(device_id_str):
        return bool(re.search(r'[a-zA-Z]', device_id_str)) and '-' in device_id_str

    # 유효한 세션만 필터링
    user_id_valid = df_clean_initial['user_id'].apply(is_numeric_user_id)
    session_id_valid = df_clean_initial['session_id'].apply(is_valid_session_id)
    device_id_valid = df_clean_initial['device_id'].apply(is_valid_device_id)
    valid_sessions_mask = user_id_valid & session_id_valid & device_id_valid
    df_cleaned = df_clean_initial[valid_sessions_mask].copy()

    # 세션 재정의
    df_redefined_sessions = df_cleaned[df_cleaned['event_key'] != '$session_end'].copy()
    df_redefined_sessions = df_redefined_sessions.sort_values(by=['user_id', 'event_datetime']).reset_index(drop=True)
    session_timeout_seconds = 20 * 60
    processed_events = []
    for user_id, user_group in df_redefined_sessions.groupby('user_id'):
        user_group = user_group.sort_values(by='event_datetime').reset_index(drop=True)
        current_session_id = user_group.loc[0, 'session_id']
        user_group['time_diff_to_next'] = user_group['event_datetime'].diff().dt.total_seconds().shift(-1)
        
        for idx, row in user_group.iterrows():
            processed_events.append(row.copy())
            if pd.notna(row['time_diff_to_next']) and row['time_diff_to_next'] >= session_timeout_seconds:
                new_session_end = row.copy()
                new_session_end['event_key'] = '$session_end'
                new_session_end['event_datetime'] = row['event_datetime']
                new_session_end['session_id'] = current_session_id
                processed_events.append(new_session_end)
                new_session_id = str(uuid.uuid4()).replace('-', '')
                next_original_event = user_group.loc[idx + 1]
                new_session_start = next_original_event.copy()
                new_session_start['event_key'] = '$session_start'
                new_session_start['event_datetime'] = next_original_event['event_datetime']
                new_session_start['session_id'] = new_session_id
                processed_events.append(new_session_start)
                current_session_id = new_session_id
            user_group.loc[idx, 'session_id'] = current_session_id

        if user_group.iloc[-1]['time_diff_to_next'] is np.nan:
            new_session_end_last = user_group.iloc[-1].copy()
            new_session_end_last['event_key'] = '$session_end'
            new_session_end_last['event_datetime'] = user_group.iloc[-1]['event_datetime']
            new_session_end_last['session_id'] = current_session_id
            processed_events.append(new_session_end_last)
            
    df_final_sessions = pd.DataFrame(processed_events)
    if 'time_diff_to_next' in df_final_sessions.columns:
        df_final_sessions = df_final_sessions.drop(columns=['time_diff_to_next'])
    df_final_sessions = df_final_sessions.sort_values(by=['user_id', 'event_datetime']).reset_index(drop=True)
    df_sorted = df_final_sessions.sort_values(["user_id", "event_datetime"])
    df_sorted["prev_event_key"] = df_sorted.groupby("user_id")["event_key"].shift(1)
    removed_df = df_sorted[df_sorted["event_key"] != df_sorted["prev_event_key"]].drop(columns="prev_event_key").reset_index(drop=True)

    print(f"이벤트 전처리 완료. 최종 데이터 크기: {removed_df.shape}")
    return removed_df

# --- 4. 사용자 요약 데이터 생성 (user_summary_script.py 통합) ---
def create_user_summary(df):
    """전처리된 이벤트를 바탕으로 사용자별 요약 데이터를 생성합니다."""
    print("--- 4. 사용자 요약 데이터 생성 시작 ---")
    event_category_mapping = {
        'launch_app': '📱 App 실행 및 세션 시작', '$session_start': '📱 App 실행 및 세션 시작',
        '$session_end': '🔚 세션 종료', 'view_home_tap': '🏠 메인/홈 진입', 'view_lab_tap': '🏠 메인/홈 진입',
        'click_bottom_navigation_lab': '🔽 하단 네비게이션 클릭', 'click_bottom_navigation_profile': '🔽 하단 네비게이션 클릭',
        'click_bottom_navigation_questions': '🔽 하단 네비게이션 클릭', 'click_bottom_navigation_timeline': '🔽 하단 네비게이션 클릭',
        'click_appbar_alarm_center': '🔼 상단 앱바 클릭', 'click_appbar_chat_rooms': '🔼 상단 앱바 클릭',
        'click_appbar_friend_plus': '🔼 상단 앱바 클릭', 'click_appbar_setting': '🔼 상단 앱바 클릭',
        'complete_signup': '🧾 회원가입/로그인 관련', 'view_login': '🧾 회원가입/로그인 관련', 'view_signup': '🧾 회원가입/로그인 관련',
        'click_purchase': '💰 하트/구매 관련', 'complete_purchase': '💰 하트/구매 관련', 'view_shop': '💰 하트/구매 관련',
        'click_copy_profile_link_ask': '❓ 질문 생성/공유/답변 등', 'click_profile_ask': '❓ 질문 생성/공유/답변 등',
        'click_question_ask': '❓ 질문 생성/공유/답변 등', 'click_question_open': '❓ 질문 생성/공유/답변 등',
        'click_question_share': '❓ 질문 생성/공유/답변 등', 'click_question_start': '❓ 질문 생성/공유/답변 등',
        'click_random_ask_normal': '❓ 질문 생성/공유/답변 등', 'click_random_ask_other': '❓ 질문 생성/공유/답변 등',
        'click_random_ask_shuffle': '❓ 질문 생성/공유/답변 등', 'complete_question': '❓ 질문 생성/공유/답변 등',
        'skip_question': '❓ 질문 생성/공유/답변 등', 'view_questions_tap': '❓ 질문 생성/공유/답변 등',
        'click_autoadd_contact': '👥 친구/소셜 기능', 'click_friend_invite': '👥 친구/소셜 기능',
        'click_invite_friend': '👥 친구/소셜 기능', 'view_friendplus_tap': '👥 친구/소셜 기능',
        'click_community_chat': '💬 채팅 기능', 'click_timeline_chat_start': '💬 채팅 기능',
        'click_notice': '🔔 알림 기능', 'click_notice_detail': '🔔 알림 기능', 'view_profile_tap': '👤 프로필/계정 관리',
        'view_timeline_tap': '📰 타임라인/피드 보기', 'button': '🛠 기타 기능/설정/유틸리티',
        'click_attendance': '🛠 기타 기능/설정/유틸리티', 'click_copy_profile_link_profile': '🛠 기타 기능/설정/유틸리티',
    }
    df['event_category'] = df['event_key'].map(event_category_mapping).fillna('기타')
    user_event_counts = df.groupby('user_id').size().reset_index(name='total_events')
    user_session_counts = df.groupby('user_id')['session_id'].nunique().reset_index(name='total_sessions')
    df['event_date'] = df['event_datetime'].dt.date
    user_active_days = df.groupby('user_id')['event_date'].nunique().reset_index(name='active_days')
    core_event_keys = ['complete_purchase', 'complete_question']
    user_core_events = df[df['event_key'].isin(core_event_keys)].groupby('user_id').size().reset_index(name='core_events')
    paying_users_id = df[df['event_key'] == 'complete_purchase']['user_id'].unique()
    all_users_id = df['user_id'].unique()
    user_payment_status = {user_id: user_id in paying_users_id for user_id in all_users_id}
    all_users_df = pd.DataFrame({'user_id': all_users_id})
    user_summary = all_users_df.merge(user_event_counts, on='user_id', how='left') \
                            .merge(user_session_counts, on='user_id', how='left') \
                            .merge(user_active_days, on='user_id', how='left') \
                            .merge(user_core_events, on='user_id', how='left')
    user_summary = user_summary.fillna(0)
    user_summary['payment_status'] = user_summary['user_id'].map(user_payment_status).fillna(False)
    
    print(f"사용자 요약 데이터 생성 완료. 총 사용자 수: {len(user_summary)}")
    return user_summary

# --- 5. 모델 로드, 예측 및 성능 평가 ---
def predict_and_evaluate(user_summary_df):
    """pkl 모델을 로드하여 결제 여부를 예측하고 성능을 평가합니다."""
    print("--- 5. 모델 예측 및 성능 평가 시작 ---")
    
    features = ['total_events', 'total_sessions', 'active_days', 'core_events']
    X = user_summary_df[features]
    y_true = user_summary_df['payment_status']

    if not os.path.exists(MODEL_PATH):
        print(f"경고: 모델 파일이 '{MODEL_PATH}'에 존재하지 않습니다. 예측 단계를 건너뜁니다.")
        return user_summary_df

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    print(f"모델 '{MODEL_PATH}' 로드 완료.")
    
    y_pred = model.predict(X)
    user_summary_df['predicted_payment'] = y_pred

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n--- 예측 성능 평가 결과 ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return user_summary_df

# --- Main 실행 함수 ---
def main(table_suffix, output_bucket):
    """전체 파이프라인을 실행하고 결과를 GCS 버킷에 저장합니다."""
    
    print(f"--- 파이프라인 실행 시작 (테이블: {table_suffix}, 버킷: {output_bucket}) ---")
    
    df_events = load_data_from_bigquery(f'{TABLE_PREFIX}_events_{table_suffix}')
    df_properties = load_data_from_bigquery(f'{TABLE_PREFIX}_properties_{table_suffix}')
    
    df_processed = preprocess_events(df_events, df_properties)
    df_summary = create_user_summary(df_processed)
    df_result = predict_and_evaluate(df_summary)
    
    # 수정: GCS 버킷에 파일 업로드
    output_filename = f'payment_prediction_result_{table_suffix}.csv'
    
    try:
        bucket = storage_client.bucket(output_bucket)
        blob = bucket.blob(output_filename)
        
        # DataFrame을 인메모리 파일로 변환하여 GCS에 직접 업로드
        df_result.to_csv(f'/tmp/{output_filename}', index=False, encoding='utf-8-sig')
        blob.upload_from_filename(f'/tmp/{output_filename}')
        
        print(f"최종 결과 데이터가 'gs://{output_bucket}/{output_filename}'에 성공적으로 저장되었습니다.")
        os.remove(f'/tmp/{output_filename}') # 임시 파일 삭제
        
    except Exception as e:
        print(f"오류: GCS 업로드 실패 - {e}")
        # 오류가 발생하더라도 태스크는 실패하지 않도록 (필요에 따라 수정 가능)
        
if __name__ == '__main__':
    # 로컬 테스트를 위한 기본값
    main('202308', 'your-gcs-bucket-name')