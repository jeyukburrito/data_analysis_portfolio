import pandas as pd
import numpy as np
import re
import uuid

# --- 1. 데이터 파일 경로 설정 ---
# 실제 파일 경로에 맞게 수정해주세요.
HACKLE_EVENTS_PATH = "C:\\Users\\Yoo\\Desktop\\codeit\\ProjectData\\Data_4\\hackle_2\\hackle_events.csv"
HACKLE_PROPERTIES_PATH = "C:\\Users\\Yoo\\Desktop\\codeit\\ProjectData\\Data_4\\hackle_2\\hackle_properties.csv"

# --- 2. 데이터 로드 ---
def load_csv_robustly(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        print(f"'{file_path}' UTF-8 인코딩으로 데이터 로드에 성공했습니다.")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949', on_bad_lines='skip')
        print(f"'{file_path}' CP949 인코딩으로 데이터 로드에 성공했습니다.")
    return df

hackle_events = load_csv_robustly(HACKLE_EVENTS_PATH)
hackle_properties = load_csv_robustly(HACKLE_PROPERTIES_PATH)

# --- 3. hackle_properties 전처리 및 session_map 생성 ---
# 문자열로 변환
hackle_properties['session_id'] = hackle_properties['session_id'].astype(str)
hackle_properties['device_id'] = hackle_properties['device_id'].astype(str)
hackle_properties['user_id'] = hackle_properties['user_id'].astype(str)

# session_id별로 user_id, device_id 매핑
session_map = hackle_properties.groupby('session_id')[['user_id', 'device_id']].first().reset_index()

# --- 4. hackle_events와 session_map 병합하여 df 생성 ---
df = hackle_events.merge(session_map, on='session_id', how='left')

# --- 5. event_datetime 형변환 ---
df['event_datetime'] = pd.to_datetime(df['event_datetime'], errors='coerce')

# --- 6. user_id가 'nan'이 아닌 행 필터링하여 df_clean (초기) 생성 ---
# 'nan' 문자열로 된 user_id를 가진 행을 제거합니다.
df_clean_initial = df[df['user_id'] != 'nan'].copy()

# --- 7. ID 규칙 검증 함수 정의 ---
def is_numeric_user_id(user_id_str):
    """user_id가 숫자로만 이루어져 있는지 확인"""
    return bool(re.fullmatch(r'\d+', user_id_str))

def is_valid_session_id(session_id_str):
    """session_id가 영어 문자를 포함하고 하이픈을 포함하지 않는지 확인"""
    return bool(re.search(r'[a-zA-Z]', session_id_str)) and '-' not in session_id_str

def is_valid_device_id(device_id_str):
    """device_id가 영어 문자를 포함하고 하이픈을 포함하는지 확인"""
    return bool(re.search(r'[a-zA-Z]', device_id_str)) and '-' in device_id_str

# --- 8. ID 규칙을 따르는 df_cleaned 생성 ---
# df_clean_initial에 규칙 적용
user_id_valid = df_clean_initial['user_id'].apply(is_numeric_user_id)
session_id_valid = df_clean_initial['session_id'].apply(is_valid_session_id)
device_id_valid = df_clean_initial['device_id'].apply(is_valid_device_id)

# 모든 규칙 만족 여부
valid_sessions_mask = user_id_valid & session_id_valid & device_id_valid

# 규칙을 따르는 세션만 남기기
df_cleaned = df_clean_initial[valid_sessions_mask].copy()
print(f"초기 데이터 {len(df)}개 중, ID 규칙을 따르는 유효한 이벤트 {len(df_cleaned)}개 추출 완료.")

# --- 9. 세션 재정의 (기존 '$session_end' 제거 및 20분 타임아웃 기준 재삽입) ---
# 기존 '$session_end' 이벤트 제거
df_redefined_sessions = df_cleaned[
    df_cleaned['event_key'] != '$session_end'
].copy()

# user_id 및 event_datetime 기준으로 정렬
df_redefined_sessions = df_redefined_sessions.sort_values(by=['user_id', 'event_datetime']).reset_index(drop=True)

# 세션 경계 이벤트 삽입 및 Session ID 재할당
session_timeout_seconds = 20 * 60 # 20분 * 60초

processed_events = []

for user_id, user_group in df_redefined_sessions.groupby('user_id'):
    user_group = user_group.sort_values(by='event_datetime').reset_index(drop=True)

    # 첫 번째 이벤트의 session_id로 시작 (원본 session_id를 사용)
    current_session_id = user_group.loc[0, 'session_id']

    # 각 이벤트의 다음 이벤트와의 시간 차이 계산
    user_group['time_diff_to_next'] = user_group['event_datetime'].diff().dt.total_seconds().shift(-1)

    for idx, row in user_group.iterrows():
        # 현재 이벤트 추가
        processed_events.append(row.copy())

        # 마지막 이벤트가 아니고, 다음 이벤트와의 시간 간격이 20분 이상인 경우
        if pd.notna(row['time_diff_to_next']) and row['time_diff_to_next'] >= session_timeout_seconds:
            # 1) '$session_end' 이벤트 추가
            new_session_end = row.copy()
            new_session_end['event_key'] = '$session_end'
            new_session_end['event_datetime'] = row['event_datetime']
            new_session_end['session_id'] = current_session_id
            processed_events.append(new_session_end)

            # 2) '$session_start' 이벤트 추가 (다음 이벤트 정보를 복사하여 생성)
            new_session_id = str(uuid.uuid4()).replace('-', '') # 새로운 session_id 생성 (하이픈 없이)

            next_original_event = user_group.loc[idx + 1]

            new_session_start = next_original_event.copy()
            new_session_start['event_key'] = '$session_start'
            new_session_start['event_datetime'] = next_original_event['event_datetime']
            new_session_start['session_id'] = new_session_id
            processed_events.append(new_session_start)

            current_session_id = new_session_id # 현재 세션 ID를 새로운 ID로 업데이트

        # 실제 user_group 내의 이벤트의 session_id를 새로운 current_session_id로 덮어씌웁니다.
        user_group.loc[idx, 'session_id'] = current_session_id

    # user_group의 마지막 이벤트 처리 (time_diff_to_next가 NaN인 경우)
    if user_group.iloc[-1]['time_diff_to_next'] is np.nan:
        new_session_end_last = user_group.iloc[-1].copy()
        new_session_end_last['event_key'] = '$session_end'
        new_session_end_last['event_datetime'] = user_group.iloc[-1]['event_datetime']
        new_session_end_last['session_id'] = current_session_id
        processed_events.append(new_session_end_last)

# 새로운 이벤트들을 포함하여 최종 데이터프레임 생성
df_final_sessions = pd.DataFrame(processed_events)

# 불필요한 임시 컬럼 제거 후 정렬
if 'time_diff_to_next' in df_final_sessions.columns:
    df_final_sessions = df_final_sessions.drop(columns=['time_diff_to_next'])

df_final_sessions = df_final_sessions.sort_values(by=['user_id', 'event_datetime']).reset_index(drop=True)
print(f"세션 재정의 완료. 최종 이벤트 수: {len(df_final_sessions)}개.")

# --- 10. 연속된 중복 event_key 제거하여 removed_df 생성 ---
# user_id별로 정렬
df_sorted = df_final_sessions.sort_values(["user_id", "event_datetime"])

# 연속된 중복 event_key 제거
df_sorted["prev_event_key"] = df_sorted.groupby("user_id")["event_key"].shift(1)
removed_df = df_sorted[df_sorted["event_key"] != df_sorted["prev_event_key"]].drop(columns="prev_event_key").reset_index(drop=True)

print(f"연속된 중복 event_key 제거 완료. 최종 'removed_df' 행 수: {len(removed_df)}개.")

# 결과 확인 (선택 사항)
print("\n'removed_df' 데이터프레임의 상위 5개 행:")
print(removed_df.head())

# removed_df를 CSV 파일로 저장 (선택 사항)
# removed_df.to_csv("event_duplicate_removed_script_output.csv", index=False, encoding="utf-8-sig")
# print("\n'removed_df'가 'event_duplicate_removed_script_output.csv' 파일로 저장되었습니다.")
