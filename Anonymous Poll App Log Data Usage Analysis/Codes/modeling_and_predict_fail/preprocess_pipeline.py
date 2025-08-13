import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from google.cloud import bigquery
from datetime import datetime
import logging

def process_bq_month_to_user_summary(month_str: str) -> pd.DataFrame:
    '''
    주어진 월(YYYYMM)에 해당하는 BigQuery 테이블 데이터를 불러와서
    이벤트 처리 → 유저 요약 → 클러스터링까지 완료한 DataFrame을 반환합니다.
    '''
    logging.info(f"📥 Loading BigQuery data for {month_str}")

    client = bigquery.Client()
    project = "codeit-final-project"
    dataset = "json_to_table"

    def load_table(name):
        table = f"{project}.{dataset}.{name}_{month_str}"
        return client.query(f"SELECT * FROM `{table}`").to_dataframe()

    df_event = load_table("hackle_events")
    df_user = load_table("user_properties")
    df_device = load_table("device_properties")
    df_hackle = load_table("hackle_properties")

    # ---------------------- 이벤트 전처리 ----------------------
    df_event = df_event.drop_duplicates(subset=["event_id"])
    df_event["event_datetime"] = pd.to_datetime(df_event["event_datetime"])
    df_event = df_event.sort_values(by=["user_id", "session_id", "event_datetime"])
    df_event["date"] = df_event["event_datetime"].dt.date

    # ---------------------- 유저 요약 생성 ----------------------
    event_summary = df_event.groupby("user_id").agg(
        total_event_count=("event_id", "count"),
        session_count=("session_id", "nunique"),
        question_count=("question_id", "nunique"),
        page_view_count=("page_name", lambda x: (x == "view").sum()),
        item_view_count=("item_name", lambda x: (x.notna()).sum()),
        heart_balance_mean=("heart_balance", "mean"),
        friend_count=("friend_count", "max"),
        votes_count=("votes_count", "max")
    ).reset_index()

    user_merged = df_user.drop_duplicates(subset=["user_id"]).merge(event_summary, on="user_id", how="left")
    user_merged = user_merged.fillna(0)

    # ---------------------- 클러스터링 ----------------------
    feature_cols = [
        "total_event_count", "session_count", "question_count",
        "page_view_count", "item_view_count", "heart_balance_mean",
        "friend_count", "votes_count"
    ]
    scaler_cols = user_merged[feature_cols].astype(float)

    kmeans = KMeans(n_clusters=5, random_state=42)
    user_merged["cluster"] = kmeans.fit_predict(scaler_cols)

    logging.info(f"✅ Finished user summary + clustering for {month_str}")

    return user_merged