import pandas as pd
import numpy as np
import re # re 모듈이 필요할 수 있으므로 추가

# --- 1. 데이터 파일 경로 설정 ---
# 이전에 생성된 'event_duplicate_removed.csv' 파일의 경로를 지정합니다.
REMOVED_DF_PATH = "C:\\Users\\Yoo\\Desktop\\codeit\\ProjectData\\Data_4\\hackle_2\\event_duplicate_removed.csv"
# accounts_paymenthistory.csv 파일 경로 (결제 유저 정보 로드용)

# --- 2. 데이터 로드 ---
def load_csv_robustly(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        print(f"'{file_path}' UTF-8 인코딩으로 데이터 로드에 성공했습니다.")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949', on_bad_lines='skip')
        print(f"'{file_path}' CP949 인코딩으로 데이터 로드에 성공했습니다.")
    return df

df = load_csv_robustly(REMOVED_DF_PATH)

# event_datetime 컬럼이 datetime 타입이 아닐 경우 변환 (로드 시 문자열로 읽힐 수 있음)
if 'event_datetime' in df.columns:
    df['event_datetime'] = pd.to_datetime(df['event_datetime'], errors='coerce')

# --- 3. event_category 매핑 정의 및 df에 적용 ---
# 최종 정리.ipynb에서 사용된 event_category_mapping을 그대로 사용합니다.
event_category_mapping = {
    # 📱 App 실행 및 세션 시작
    'launch_app': '📱 App 실행 및 세션 시작',
    '$session_start': '📱 App 실행 및 세션 시작',

    # 🔚 세션 종료
    '$session_end': '🔚 세션 종료',

    # 🏠 메인/홈 진입
    'view_home_tap': '🏠 메인/홈 진입',
    'view_lab_tap': '🏠 메인/홈 진입',

    # 🔽 하단 네비게이션 클릭
    'click_bottom_navigation_lab': '🔽 하단 네비게이션 클릭',
    'click_bottom_navigation_profile': '🔽 하단 네비게이션 클릭',
    'click_bottom_navigation_questions': '🔽 하단 네비게이션 클릭',
    'click_bottom_navigation_timeline': '🔽 하단 네비게이션 클릭',

    # 🔼 상단 앱바 클릭
    'click_appbar_alarm_center': '🔼 상단 앱바 클릭',
    'click_appbar_chat_rooms': '🔼 상단 앱바 클릭',
    'click_appbar_friend_plus': '🔼 상단 앱바 클릭',
    'click_appbar_setting': '🔼 상단 앱바 클릭',

    # 🧾 회원가입/로그인 관련
    'complete_signup': '🧾 회원가입/로그인 관련',
    'view_login': '🧾 회원가입/로그인 관련',
    'view_signup': '🧾 회원가입/로그인 관련',

    # 💰 하트/구매 관련
    'click_purchase': '💰 하트/구매 관련',
    'complete_purchase': '💰 하트/구매 관련',
    'view_shop': '💰 하트/구매 관련',

    # ❓ 질문 생성/공유/답변 등
    'click_copy_profile_link_ask': '❓ 질문 생성/공유/답변 등',
    'click_profile_ask': '❓ 질문 생성/공유/답변 등',
    'click_question_ask': '❓ 질문 생성/공유/답변 등',
    'click_question_open': '❓ 질문 생성/공유/답변 등',
    'click_question_share': '❓ 질문 생성/공유/답변 등',
    'click_question_start': '❓ 질문 생성/공유/답변 등',
    'click_random_ask_normal': '❓ 질문 생성/공유/답변 등',
    'click_random_ask_other': '❓ 질문 생성/공유/답변 등',
    'click_random_ask_shuffle': '❓ 질문 생성/공유/답변 등',
    'complete_question': '❓ 질문 생성/공유/답변 등',
    'skip_question': '❓ 질문 생성/공유/답변 등',
    'view_questions_tap': '❓ 질문 생성/공유/답변 등',

    # 👥 친구/소셜 기능
    'click_autoadd_contact': '👥 친구/소셜 기능',
    'click_friend_invite': '👥 친구/소셜 기능',
    'click_invite_friend': '👥 친구/소셜 기능',
    'view_friendplus_tap': '👥 친구/소셜 기능',

    # 💬 채팅 기능
    'click_community_chat': '💬 채팅 기능',
    'click_timeline_chat_start': '💬 채팅 기능',

    # 🔔 알림 기능
    'click_notice': '🔔 알림 기능',
    'click_notice_detail': '🔔 알림 기능',

    # 👤 프로필/계정 관리
    'view_profile_tap': '👤 프로필/계정 관리',

    # 📰 타임라인/피드 보기
    'view_timeline_tap': '📰 타임라인/피드 보기',

    # 🛠 기타 기능/설정/유틸리티
    'button': '🛠 기타 기능/설정/유틸리티',
    'click_attendance': '🛠 기타 기능/설정/유틸리티',
    'click_copy_profile_link_profile': '🛠 기타 기능/설정/유틸리티',
}

df['event_category'] = df['event_key'].map(event_category_mapping)
print("데이터프레임 'df'의 'event_category' 컬럼이 업데이트되었습니다.")

# 매핑되지 않은 event_key가 있다면 '기타' 카테고리로 처리 (선택 사항)
df['event_category'] = df['event_category'].fillna('기타')
print("매핑되지 않은 event_key는 '기타' 카테고리로 처리되었습니다.")

# --- 4. user_summary 데이터프레임 생성 로직 (수정된 버전) ---
print("\n'user_summary' 데이터프레임 생성 시작 (수정된 로직 적용)...")

# 4.1. user_event_counts: 사용자별 총 이벤트 수
user_event_counts = df.groupby('user_id').size().reset_index(name='total_events')

# 4.2. user_session_counts: 사용자별 총 세션 수
user_session_counts = df.groupby('user_id')['session_id'].nunique().reset_index(name='total_sessions')

# 4.3. user_active_days: 사용자별 활동 일수
# event_datetime에서 날짜만 추출하여 고유한 날짜의 수를 셉니다.
df['event_date'] = df['event_datetime'].dt.date
user_active_days = df.groupby('user_id')['event_date'].nunique().reset_index(name='active_days')

# 4.4. user_core_events: 사용자별 핵심 이벤트 수
# '핵심 이벤트'는 'complete_purchase', 'complete_question' 등으로 정의될 수 있습니다.
# 여기서는 '최종 정리.ipynb'에서 'core_events'가 어떻게 정의되었는지 명확하지 않으므로,
# 예시로 'complete_purchase'와 'complete_question'을 핵심 이벤트로 가정합니다.
# 실제 노트북에서 사용된 'core_events' 정의를 확인하여 수정해야 합니다.
core_event_keys = ['complete_purchase', 'complete_question'] # 예시
user_core_events = df[df['event_key'].isin(core_event_keys)].groupby('user_id').size().reset_index(name='core_events')

# 4.5. user_payment: 결제 유저 정보 (사용자 제공 로직으로 변경)
print("\n--- 1. 사용자별 결제 여부 정의 ---")
# 'complete_purchase' 이벤트를 발생시킨 user_id 추출
paying_users_id = df[df['event_key'] == 'complete_purchase']['user_id'].unique()

# 전체 user_id 목록
all_users_id = df['user_id'].unique()

# 결제 상태를 True/False로 매핑한 딕셔너리 생성
user_payment_status = {user_id: user_id in paying_users_id for user_id in all_users_id}

print(f"총 사용자 수: {len(all_users_id)}명")
print(f"결제 유저 수: {len(paying_users_id)}명")
print(f"비결제 유저 수: {len(all_users_id) - len(paying_users_id)}명")
print("--- 1. 사용자별 결제 여부 정의 완료 ---")


# 모든 user_id를 포함하는 기본 데이터프레임 생성 (결제 여부 매핑을 위해)
all_users_df = pd.DataFrame({'user_id': all_users_id})

# user_summary 병합
user_summary = all_users_df.merge(user_event_counts, on='user_id', how='left') \
                        .merge(user_session_counts, on='user_id', how='left') \
                        .merge(user_active_days, on='user_id', how='left') \
                        .merge(user_core_events, on='user_id', how='left')

# 결측값 처리 (이벤트가 없는 사용자는 0으로 채움)
user_summary = user_summary.fillna(0)

# payment_status 추가 (사용자 제공 로직으로 변경)
user_summary['payment_status'] = user_summary['user_id'].map(user_payment_status).fillna(False)
user_summary['payment_status_str'] = user_summary['payment_status'].apply(lambda x: '결제 유저' if x else '비결제 유저')

print("'user_summary' 데이터프레임 생성 완료.")

# 결과 출력
print(f"총 사용자 수: {len(user_summary)}명")
print(f"결제 유저 수: {user_summary['payment_status'].sum()}명")
print(f"비결제 유저 수: {len(user_summary) - user_summary['payment_status'].sum()}명")

# 군집화에 사용할 특성 정의 (user_summary 생성 후)
features_for_clustering = ['total_events', 'total_sessions', 'active_days', 'core_events']
X_for_clustering = user_summary[features_for_clustering].copy()

print("\n군집화에 사용할 특성 데이터 (X_for_clustering) 준비 완료.")
print(X_for_clustering.head())

# user_summary를 CSV 파일로 저장 (선택 사항)
user_summary.to_csv("user_summary_for_clustering.csv", index=False, encoding="utf-8-sig")
print("\n'user_summary_for_clustering.csv' 파일로 저장되었습니다.")