import ijson
from google.cloud import storage, bigquery
import pandas as pd
import gc

def process_monthly_json_to_bq(bucket_name, folder_path, bq_dataset, bq_project):
    storage_client = storage.Client()
    bq_client = bigquery.Client(project=bq_project)

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_path))

    print(f"[INFO] Found {len(blobs)} files in {folder_path}")

    table_suffix = folder_path.strip('/').replace('-', '')

    for blob in blobs:
        if not blob.name.endswith(".json"):
            print(f"[INFO] Skipping non-json file: {blob.name}")
            continue

        print(f"[INFO] Processing file: {blob.name}")

        hackle_props = []
        device_props = []
        hackle_events = []
        user_props = []

        try:
            with blob.open("r") as f:
                objects = ijson.items(f, "item")

                for log in objects:
                    props = log.get("hackle_properties", {})
                    user_p = log.get("user_properties", {})
                    event_props = log.get("event_properties", {})

                    hackle_props.append({
                        "session_id": log.get("session_id"),
                        "user_id": log.get("user_id"),
                        "device_id": log.get("device_id"),
                        "language": props.get("language"),
                        "osname": props.get("osname"),
                        "osversion": props.get("osversion"),
                        "versionname": props.get("versionname")
                    })

                    device_props.append({
                        "device_id": log.get("device_id"),
                        "device_model": props.get("devicemodel"),
                        "device_vendor": props.get("devicevendor")
                    })

                    hackle_events.append({
                        "event_id": log.get("id"),
                        "event_datetime": log.get("Asia/Seoul"),
                        "event_key": log.get("event_key"),
                        "session_id": log.get("session_id"),
                        "item_name": event_props.get("item_name"),
                        "page_name": event_props.get("page_name"),
                        "question_id": event_props.get("question_id")
                    })

                    user_props.append({
                        "user_id": log.get("user_id"),
                        "class": user_p.get("class"),
                        "gender": user_p.get("gender"),
                        "grade": user_p.get("grade"),
                        "school_id": user_p.get("school_id"),
                        "friend_count": user_p.get("friend_count"),
                        "votes_count": user_p.get("votes_count"),
                        "heart_balance": user_p.get("heart_balance"),
                    })

            # 파일 단위 DataFrame 변환 및 업로드
            def upload_to_bq(df, table_name):
                if df.empty:
                    print(f"[INFO] {table_name} is empty, skipping upload.")
                    return

                table_id = f"{bq_project}.{bq_dataset}.{table_name}_{table_suffix}"
                job = bq_client.load_table_from_dataframe(df.drop_duplicates(), table_id)
                job.result()
                print(f"[INFO] Uploaded {len(df)} rows to {table_id}")

            upload_to_bq(pd.DataFrame(hackle_props), "hackle_properties")
            upload_to_bq(pd.DataFrame(device_props), "device_properties")
            upload_to_bq(pd.DataFrame(hackle_events), "hackle_events")
            upload_to_bq(pd.DataFrame(user_props), "user_properties")

        except Exception as e:
            print(f"[ERROR] Failed to process {blob.name}: {e}")

        finally:
            del hackle_props, device_props, hackle_events, user_props
            gc.collect()

    print(f"[INFO] Completed processing month folder: {folder_path}")