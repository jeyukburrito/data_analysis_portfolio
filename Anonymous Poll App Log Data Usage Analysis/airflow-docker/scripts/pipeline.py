
import os
import logging
import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, precision_recall_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb

# --- 기본 설정 ---
BASE_DIR = os.getenv('AIRFLOW_HOME', '/opt/airflow-docker')
DATA_ROOT = os.path.join(BASE_DIR, "data")
RESULTS_ROOT = os.path.join(BASE_DIR, "results")
PROCESSED_DATA_PATH = os.path.join(RESULTS_ROOT, "modeling_table.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def preprocessing_task():
    """5개의 원본 테이블을 모두 사용하여 전체 전처리 과정을 수행합니다."""
    logging.info("=== TASK 1: 데이터 전처리 파이프라인 시작 (5개 테이블 사용) ===")
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # --- 1. 5개 테이블 모두 로드 ---
    logging.info("5개 원본 소스 데이터 로드 중...")
    try:
        user_df = pd.read_csv(os.path.join(DATA_ROOT, "accounts_user.csv"), encoding='utf-8-sig')
        school_df = pd.read_csv(os.path.join(DATA_ROOT, "accounts_school.csv"), encoding='utf-8-sig')
        user_props_df = pd.read_csv(os.path.join(DATA_ROOT, "user_properties.csv"), encoding='utf-8-sig', low_memory=False)
        hackle_props_df = pd.read_csv(os.path.join(DATA_ROOT, "hackle_properties.csv"), encoding='utf-8-sig', low_memory=False)
        user_summary_df = pd.read_csv(os.path.join(DATA_ROOT, "user_summary.csv"), encoding='cp949')
    except FileNotFoundError as e:
        logging.error(f"오류: 파일을 찾을 수 없습니다. {e}")
        raise

    # --- 2. ID 컬럼명 통일 및 타입 변환 ---
    logging.info("병합 키(user_id, school_id)의 이름 및 데이터 타입을 통일합니다.")
    user_df.rename(columns={'id': 'user_id'}, inplace=True)
    school_df.rename(columns={'id': 'school_id'}, inplace=True)
    user_props_df.rename(columns={'id': 'user_id'}, inplace=True, errors='ignore')

    dfs_to_clean = [user_df, user_props_df, hackle_props_df, user_summary_df]
    for df in dfs_to_clean:
        if 'user_id' in df.columns:
            df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
            df.dropna(subset=['user_id'], inplace=True)
            df['user_id'] = df['user_id'].astype('int64')

    # --- 3. accounts_user.csv 리스트 컬럼 전처리 ---
    list_cols = {
        'friend_id_list': 'friend_id_count',
        'block_user_id_list': 'block_user_id_count',
        'hide_user_id_list': 'hide_user_id_count'
    }
    for src, dst in list_cols.items():
        if src in user_df.columns:
            user_df[dst] = user_df[src].apply(lambda x: len(ast.literal_eval(x)) if pd.notna(x) and x.startswith('[') else 0)
            user_df.drop(columns=[src], inplace=True)

    # --- 4. 데이터 병합 ---
    logging.info("데이터 병합 시작...")
    merged_df = pd.merge(user_summary_df, user_df, on='user_id', how='inner')
    user_props_latest = user_props_df.drop_duplicates(subset=['user_id'], keep='last')
    merged_df = pd.merge(merged_df, user_props_latest, on='user_id', how='left', suffixes=('', '_user_props'))
    hackle_props_latest = hackle_props_df.drop_duplicates(subset=['user_id'], keep='last')
    merged_df = pd.merge(merged_df, hackle_props_latest, on='user_id', how='left', suffixes=('', '_hackle_props'))
    merged_df.dropna(subset=['school_id'], inplace=True)
    merged_df['school_id'] = merged_df['school_id'].astype(int)
    merged_df = pd.merge(merged_df, school_df, on='school_id', how='left', suffixes=('', '_school'))

    # --- 5. 데이터 클리닝 및 피처 엔지니어링 ---
    logging.info("데이터 클리닝 및 피처 엔지니어링 중...")
    merged_df = merged_df[merged_df['school_id'] != 1].copy()
    merged_df.dropna(subset=['address', 'student_count', 'school_type'], inplace=True)
    merged_df['pending_chat'] = merged_df['pending_chat'].clip(lower=0)

    modeling_table = merged_df.copy()
    modeling_table['school_grade'] = modeling_table['school_type'].astype(str) + modeling_table['grade'].astype(str)
    
    address_dict = {
        '강원도': 'Gangwon', '경기도': 'Gyeonggi', '경기': 'Gyeonggi', '경상북도': 'Gyeongbuk', '경북': 'Gyeongbuk',
        '경상남도': 'Gyeongnam', '경남': 'Gyeongnam', '충청남도': 'Chungnam', '충남': 'Chungnam', '충청북도': 'Chungbuk',
        '광주광역시': 'Gwangju', '대구광역시': 'Daegu', '대구': 'Daegu', '대전광역시': 'Daejeon', '부산광역시': 'Busan',
        '울산광역시': 'Ulsan', '인천광역시': 'Incheon', '인천': 'Incheon', '서울특별시': 'Seoul', '서울': 'Seoul',
        '세종특별자치시': 'Sejong', '제주특별자치도': 'Jeju', '전라남도': 'Jeonnam', '전라북도': 'Jeonbuk', '전북': 'Jeonbuk',
    }
    modeling_table.loc[modeling_table['address'] == '대한민국 강원도 철원군', 'address'] = '강원도 철원군'
    modeling_table.loc[modeling_table['address'] == '충청남도 세종특별자치시', 'address'] = '세종특별자치시'
    modeling_table['province'] = modeling_table['address'].str.split().str[0].map(address_dict)

    drop_cols = [
        'user_id', 'payment_status_str', 'is_superuser', 'is_staff', 'point', 'created_at', 'group_id', 
        'school_id', 'student_count', 'group_class_num', 'school_type', 'grade', 'address',
        'group_school_id', 'session_id', 'language', 'osversion', 'versionname', 'device_id',
        'class_user_props', 'gender_user_props', 'grade_user_props', 'school_id_user_props',
        'session_id_hackle_props', 'language_hackle_props', 'osname_hackle_props', 'osversion_hackle_props', 'versionname_hackle_props', 'device_id_hackle_props',
        'address_school', 'student_count_school', 'school_type_school', 'id', 'class'
    ]
    modeling_table.drop(columns=[col for col in drop_cols if col in modeling_table.columns], inplace=True)

    # --- 6. 최종 테이블 저장 ---
    logging.info(f"전처리 완료. 최종 테이블 저장: {PROCESSED_DATA_PATH}")
    modeling_table.to_csv(PROCESSED_DATA_PATH, index=False, encoding='cp949')
    logging.info(f"최종 데이터 Shape: {modeling_table.shape}")
    logging.info("=== TASK 1: 데이터 전처리 파이프라인 종료 ===")


def train_evaluate_task():
    """모델을 학습/평가하고, 주요 지표가 담긴 딕셔너리를 반환합니다."""
    logging.info("=== TASK 2: 모델 학습 및 평가 파이프라인 시작 ===")
    
    logging.info(f"전처리된 데이터 로드: {PROCESSED_DATA_PATH}")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, encoding='cp949')
    except FileNotFoundError:
        logging.error(f"{PROCESSED_DATA_PATH} 파일을 찾을 수 없습니다. 전처리 Task를 먼저 실행해야 합니다.")
        raise

    logging.info("VIF 기반 피처 선택 시작...")
    TARGET = 'payment_status'
    X = df.drop(columns=[TARGET])
    numeric_cols_for_vif = X.select_dtypes(include=np.number).columns.tolist()
    
    features = numeric_cols_for_vif[:]
    while True:
        X_numeric = X[features].assign(const=1)
        vif_data = pd.DataFrame({
            'feature': X_numeric.columns,
            'VIF': [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
        })
        max_vif = vif_data.loc[vif_data['feature']!='const','VIF'].max()
        if max_vif <= 10.0:
            break
        drop_feat = vif_data.loc[vif_data['VIF']==max_vif,'feature'].iloc[0]
        features.remove(drop_feat)
        logging.info(f"VIF > 10: '{drop_feat}' 제거 (VIF: {max_vif:.2f})")

    final_features = features + X.select_dtypes(exclude=np.number).columns.tolist()
    logging.info(f"최종 선택된 피처: {final_features}")
    
    X = df[final_features]
    y = df[TARGET].astype(int)

    if 'cluster' in X.columns:
        X['cluster'] = X['cluster'].astype('category')

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    ordinal_cols = ['school_grade']
    nominal_cols = [col for col in cat_cols if col not in ordinal_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols)
    ], remainder='passthrough')
    
    lgbm = lgb.LGBMClassifier(random_state=42)
    pipeline = Pipeline([('preprocessor', preprocessor), ('sampler', ADASYN(random_state=42)), ('classifier', lgbm)])
    
    logging.info("Baseline LightGBM 모델 학습 시작...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    best_thresh_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_thresh_idx]
    y_pred_optim = (y_prob >= best_thresh).astype(int)

    # 평가 지표 계산
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred_optim, target_names=['Non-Payer', 'Payer'])
    
    logging.info("모델 평가 완료.")

    # XCom으로 전달할 결과 딕셔너리 생성
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_thresh,
        "classification_report": report
    }
    
    return metrics


if __name__ == '__main__':
    logging.info("=== AIRFLOW PIPELINE SCRIPT START ===")
    preprocessing_task()
    train_evaluate_task()
    logging.info("=== AIRFLOW PIPELINE SCRIPT END ===")
