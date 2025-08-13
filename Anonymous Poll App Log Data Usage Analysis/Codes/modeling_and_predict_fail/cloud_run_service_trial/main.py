import os
import pickle
import pandas as pd
import gcsfs
from flask import Flask, request, jsonify

# Flask 웹 서버 생성
app = Flask(__name__)

# --- GCS에서 모델을 한번만 로드하기 위한 설정 ---
# 1. GCS 버킷에 .pkl 모델을 미리 업로드해야 합니다.
MODEL_URI = "gs://pkl_bucket/models/payment_prediction_model.pkl"
fs = gcsfs.GCSFileSystem()
pipeline = None

def load_model():
    """GCS에서 .pkl 모델을 로드하는 함수"""
    global pipeline
    if pipeline is None:
        with fs.open(MODEL_URI, 'rb') as f:
            pipeline = pickle.load(f)
    return pipeline

# --- BigQuery가 호출할 API 엔드포인트 ---
@app.route('/', methods=['POST'])
def predict():
    """BigQuery로부터 데이터를 받아 예측을 수행합니다."""
    # BigQuery가 보낸 요청 데이터 파싱
    request_json = request.get_json()
    calls = request_json['calls'] # BigQuery는 데이터를 'calls'라는 키에 담아 보냅니다.

    # 모델 로드
    model_pipeline = load_model()
    
    replies = []
    
    for call in calls:
        # BigQuery에서 전달된 데이터의 순서에 맞춰 Pandas DataFrame 생성
        # 이 피처 목록과 순서는 .pkl 모델 훈련 시 사용했던 것과 정확히 일치해야 합니다.
        features = [
            'total_events', 'total_sessions', 'active_days', 'core_events', 
            'school_grade', 'osname', 'friend_id_count', 'gender', 'alarm_count', 
            'report_count', 'pending_votes', 'ban_status', 'is_push_on', 'class', 
            'hide_user_id_count', 'block_user_id_count', 'province', 'cluster', 
            'pending_chat', 'user_props_id'
        ]
        df = pd.DataFrame([call], columns=features)

        # .pkl 모델을 사용해 예측 (예: 0 또는 1)
        prediction = model_pipeline.predict(df)[0]
        replies.append(prediction)

    # BigQuery에 예측 결과 목록을 JSON 형태로 반환
    return jsonify({"replies": replies})

# --- 서버 실행 ---
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))