import json
import os

def split_local_json_file(file_path):
    """
    로컬에 저장된 JSON 배열 파일을 절반으로 나눠서 
    같은 디렉토리에 -1, -2 접미어를 붙여 저장합니다.

    예: 2023-09-15.json → 2023-09-15-1.json, 2023-09-15-2.json
    """
    # 파일 이름 분해
    dir_path, filename = os.path.split(file_path)
    base_name = filename.replace(".json", "")
    
    # JSON 로드
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    midpoint = total // 2

    part1 = data[:midpoint]
    part2 = data[midpoint:]

    # 저장 경로
    part1_path = os.path.join(dir_path, f"{base_name}-1.json")
    part2_path = os.path.join(dir_path, f"{base_name}-2.json")

    with open(part1_path, "w", encoding="utf-8") as f:
        json.dump(part1, f, ensure_ascii=False, indent=2)

    with open(part2_path, "w", encoding="utf-8") as f:
        json.dump(part2, f, ensure_ascii=False, indent=2)

    print(f"✅ 분할 완료: {filename} → {os.path.basename(part1_path)}, {os.path.basename(part2_path)}")
    print(f"  총 {total}개 → {len(part1)} / {len(part2)}")

# 예시: 실제로 사용할 파일 경로를 아래에 입력하세요
split_local_json_file("C:\\Users\\PC\\Codeit_sprint_DA_6_final_project\\빅쿼리업로드\\2023-09-15.json")
split_local_json_file("C:\\Users\\PC\\Codeit_sprint_DA_6_final_project\\빅쿼리업로드\\2023-09-21.json")
split_local_json_file("C:\\Users\\PC\\Codeit_sprint_DA_6_final_project\\빅쿼리업로드\\2023-09-27.json")