# -*- coding: utf-8 -*-
"""
모델 학습 및 저장 스크립트 (train_model.py)

이 스크립트는 전처리된 데이터를 사용하여 LightGBM 모델을 학습시키고,
학습된 전체 파이프라인(전처리기, 오버샘플러, 모델 포함)을 pickle 파일로 저장합니다.

주요 기능:
1. VIF(분산 팽창 계수)를 이용한 다중공선성 문제 해결 및 피처 선택
2. 범주형/수치형 데이터에 대한 전처리 파이프라인(ColumnTransformer) 구성
3. ADASYN을 사용한 데이터 불균형 문제 처리
4. LightGBM 모델 학습
5. 학습된 최종 파이프라인을 'lgbm_pipeline.pkl' 파일로 저장

실행 전 필요사항:
- 'pipeline.py'의 'preprocessing_task'가 실행되어 'results/modeling_table.csv' 파일이 생성되어 있어야 합니다.
"""
import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb

# --- 경로 및 로깅 설정 ---
# 이 스크립트 파일의 위치를 기준으로 기본 경로를 설정합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# 결과 및 모델을 저장할 폴더 경로를 설정합니다.
RESULTS_ROOT = os.path.join(BASE_DIR, "results")
MODELS_ROOT = os.path.join(BASE_DIR, "models")
# 전처리된 데이터와 최종 모델 파일의 전체 경로를 정의합니다.
PROCESSED_DATA_PATH = os.path.join(RESULTS_ROOT, "modeling_table.csv")
MODEL_PATH = os.path.join(MODELS_ROOT, "lgbm_pipeline.pkl")

# 로깅 기본 설정: 시간, 로그 레벨, 메시지를 포함하여 출력
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train_and_save_model():
    """
    모델을 학습하고, 학습된 파이프라인 전체를 Pickle 파일로 저장합니다.
    """
    logging.info("=== 모델 학습 및 저장 파이프라인 시작 ===")
    # 모델을 저장할 'models' 폴더가 없으면 생성합니다.
    os.makedirs(MODELS_ROOT, exist_ok=True)

    logging.info(f"전처리된 데이터 로드 시작: {PROCESSED_DATA_PATH}")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, encoding='cp949')
    except FileNotFoundError:
        logging.error(f"오류: {PROCESSED_DATA_PATH} 파일을 찾을 수 없습니다. 'pipeline.py'의 전처리 Task를 먼저 실행해야 합니다.")
        raise

    # --- VIF(분산 팽창 계수) 기반 피처 선택 ---
    # 다중공선성 문제를 완화하기 위해 VIF가 높은 피처를 순차적으로 제거합니다.
    logging.info("VIF 기반 피처 선택 시작...")
    TARGET = 'payment_status'
    X = df.drop(columns=[TARGET])
    # VIF 계산은 수치형 데이터에 대해서만 수행합니다.
    numeric_cols_for_vif = X.select_dtypes(include=np.number).columns.tolist()
    
    features = numeric_cols_for_vif[:]
    while True:
        # 상수항을 추가하여 VIF를 계산합니다.
        X_numeric = X[features].assign(const=1)
        vif_data = pd.DataFrame({
            'feature': X_numeric.columns,
            'VIF': [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
        })
        # 상수항을 제외한 피처 중 가장 높은 VIF 값을 찾습니다.
        max_vif = vif_data.loc[vif_data['feature']!='const','VIF'].max()
        
        # 모든 피처의 VIF가 10 이하이면 반복을 중단합니다.
        if max_vif <= 10.0:
            break
        
        # 가장 높은 VIF를 가진 피처를 제거합니다.
        drop_feat = vif_data.loc[vif_data['VIF']==max_vif,'feature'].iloc[0]
        features.remove(drop_feat)
        logging.info(f"VIF > 10: '{drop_feat}' 피처 제거 (VIF: {max_vif:.2f})")

    # VIF를 통해 선택된 수치형 피처와 모든 범주형 피처를 합쳐 최종 피처 목록을 만듭니다.
    final_features = features + X.select_dtypes(exclude=np.number).columns.tolist()
    logging.info(f"최종 선택된 피처 목록: {final_features}")
    
    X = df[final_features]
    y = df[TARGET].astype(int)

    # 'cluster' 컬럼이 있다면 범주형으로 타입을 명시적으로 변환합니다.
    if 'cluster' in X.columns:
        X['cluster'] = X['cluster'].astype('category')

    # --- 데이터 분할 및 전처리 파이프라인 구축 ---
    # 피처 유형에 따라 다른 전처리 방법을 적용하기 위해 컬럼 목록을 정의합니다.
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    ordinal_cols = ['school_grade']  # 순서가 있는 범주형 피처
    nominal_cols = [col for col in cat_cols if col not in ordinal_cols] # 순서가 없는 명목형 피처

    # 모델 학습을 위해 데이터를 학습용(train)과 테스트용(test)으로 분리합니다.
    # 여기서는 전체 데이터를 사용하여 모델을 학습시키므로, X_test, y_test는 사용하지 않습니다.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ColumnTransformer: 각기 다른 피처 그룹에 다른 변환을 적용하는 파이프라인
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),  # 수치형 피처: 표준화
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols), # 순서형 피처: 순서 인코딩
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols) # 명목형 피처: 원-핫 인코딩
    ], remainder='passthrough') # 나머지 피처는 그대로 통과
    
    # --- 모델 학습 파이프라인 구축 및 실행 ---
    lgbm = lgb.LGBMClassifier(random_state=42)
    # 최종 파이프라인: 전처리 -> 오버샘플링(ADASYN) -> 모델(LightGBM)
    pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('sampler', ADASYN(random_state=42)), 
        ('classifier', lgbm)
    ])
    
    logging.info("최종 LightGBM 모델 파이프라인 학습 시작...")
    pipeline.fit(X_train, y_train)
    
    # --- 학습된 파이프라인 저장 ---
    logging.info(f"모델 학습 완료. 파이프라인 저장 위치: {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    logging.info("=== 모델 학습 및 저장 파이프라인 성공적으로 종료 ===")

if __name__ == '__main__':
    # 이 스크립트가 직접 실행될 때 train_and_save_model 함수를 호출합니다.
    train_and_save_model()