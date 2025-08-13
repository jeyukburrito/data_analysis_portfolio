# 1. 베이스가 될 파이썬 환경 선택
FROM python:3.9-slim

# 2. 컨테이너 내부의 작업 폴더 설정
WORKDIR /app

# 3. 필요한 라이브러리 먼저 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

# 5. 서버 실행 명령어 (gunicorn 사용)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]