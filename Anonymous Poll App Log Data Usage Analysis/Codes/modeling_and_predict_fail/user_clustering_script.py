import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib # 한글 폰트 설정을 위해 필요
from yellowbrick.cluster import KElbowVisualizer # KElbowVisualizer 임포트

# --- 1. user_summary.csv 파일 경로 설정 ---
# user_summary.csv 파일의 경로를 지정합니다.
USER_SUMMARY_PATH = "C:\\Users\\Yoo\\Desktop\\codeit\\user_summary_for_clustering.csv" # 이 경로는 실제 user_summary.csv 파일 경로로 수정해야 합니다.

# --- 2. user_summary.csv 로드 ---
def load_csv_robustly(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        print(f"'{file_path}' UTF-8 인코딩으로 데이터 로드에 성공했습니다.")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949', on_bad_lines='skip')
        print(f"'{file_path}' CP949 인코딩으로 데이터 로드에 성공했습니다.")
    return df

user_summary = load_csv_robustly(USER_SUMMARY_PATH)

print(f"'{USER_SUMMARY_PATH}' 로드 완료. 데이터 크기: {user_summary.shape}")

# user_id를 인덱스로 설정 (군집 분석에 사용되지 않는 컬럼)
if 'user_id' in user_summary.columns:
    user_summary_indexed = user_summary.set_index('user_id')
else:
    user_summary_indexed = user_summary.copy()
    print("경고: 'user_id' 컬럼이 없습니다. 인덱스 설정 없이 진행합니다.")

# --- 군집화 준비 ---
# 군집화에 사용할 활동량 지표 선택
features_for_clustering = ['total_events', 'total_sessions', 'active_days', 'core_events']
X = user_summary_indexed[features_for_clustering].copy() # user_summary_indexed에서 특성 선택

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_clustering, index=X.index)

print("\n--- 최적의 K (군집 개수) 탐색 (엘보우 방법) ---")
# 시각화를 위한 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

model = KMeans(random_state=42, n_init='auto') # n_init='auto'는 최신 버전 Kmeans 권장사항
visualizer = KElbowVisualizer(model, k=(2,10), metric='distortion', timings=False) # k 범위를 2~9로 설정 (10은 포함 안됨)

plt.figure(figsize=(8, 6))
visualizer.fit(X_scaled)
visualizer.show()
print("-> 엘보우 차트에서 꺾이는 지점을 찾아 최적의 K를 결정하세요. (가장 적절한 K값을 아래 `optimal_k`에 직접 입력하세요.)")

# 사용자가 직접 최적의 K 값을 입력 (엘보우 차트를 보고 결정)
optimal_k = 3 # 예시: 엘보우 차트를 보고 3이 적절하다고 가정 (수정 가능)
print(f"\n결정된 최적의 K (군집 개수): {optimal_k}")


# --- K-Means 군집화 실행 ---
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
user_summary['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n--- K-Means 군집화 완료 ({optimal_k}개 군집 생성) ---")


# --- 군집별 특성 분석 ---
print("\n--- 군집별 특성 분석 ---")

# 각 군집의 사용자 수
cluster_user_counts = user_summary['cluster'].value_counts().sort_index()
print("\n 군집별 사용자 수:\n", cluster_user_counts)

# 각 군집의 평균 활동량 지표
cluster_activity_summary = user_summary.groupby('cluster')[features_for_clustering].mean()
print("\n 군집별 평균 활동량 지표:")
print(cluster_activity_summary.round(2))

# 각 군집별 결제율 계산
cluster_payment_rates = user_summary.groupby('cluster')['payment_status'].mean() * 100
print("\n군집별 결제율 (%):\n", cluster_payment_rates.round(2))

# 군집별 특성 통합 (편의성을 위해)
cluster_analysis_df = cluster_activity_summary.copy()
cluster_analysis_df['user_count'] = cluster_user_counts
cluster_analysis_df['payment_rate'] = cluster_payment_rates
cluster_analysis_df = cluster_analysis_df.sort_values(by='payment_rate', ascending=False) # 결제율 높은 순으로 정렬

print("\n--- 통합 군집 분석 결과 (결제율 높은 순 정렬) ---")
print(cluster_analysis_df.round(2).to_string())

# --- 군집별 결제율 및 활동량 특징 시각화 ---
print("\n--- 군집별 결과 시각화 ---")

# 군집별 결제율 막대 그래프
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_payment_rates.index, y=cluster_payment_rates.values, palette='viridis')
plt.title("군집별 결제율", fontsize=16)
plt.xlabel("군집 번호", fontsize=12)
plt.ylabel("결제율 (%)", fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 군집별 활동량 지표 시각화 (Normalized values for comparison)
# 각 지표별로 가장 높고 낮은 군집을 시각적으로 비교하기 위함
cluster_activity_scaled_mean = pd.DataFrame(scaler.transform(cluster_activity_summary),
                                            columns=features_for_clustering,
                                            index=cluster_activity_summary.index)
cluster_activity_scaled_mean = cluster_activity_scaled_mean.reset_index().melt(id_vars='cluster', var_name='feature', value_name='scaled_value')

plt.figure(figsize=(12, 7))
sns.barplot(data=cluster_activity_scaled_mean, x='feature', y='scaled_value', hue='cluster', palette='coolwarm')
plt.title("군집별 활동량 지표 평균 (스케일링 적용)", fontsize=16)
plt.xlabel("활동량 지표", fontsize=12)
plt.ylabel("평균값 (스케일링)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='군집', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\n### 군집화 분석 완료 ###")

user_summary.to_csv('user_cluster.csv', index=False, encoding='cp949') # 모델링용 테이블에 결합하기 위해 user_summary 데이터 프레임 저장  -> 클러스터링 이후, 데이터만 저장하는 로직만 남기자