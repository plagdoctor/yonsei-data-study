# ============================================================
# [Step 2] 데이터 불러오기
# ============================================================
# Google Drive의 Data 폴더에 저장된
# 교통사고 데이터셋(dataset_traffic_accident.csv)을 불러옵니다.
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# 데이터 분석 라이브러리 임포트
import pandas as pd

# 데이터 파일 경로 설정
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident.csv'

# CSV 파일 불러오기
df = pd.read_csv(data_path)

# 데이터 기본 정보 확인
print("=" * 60)
print("데이터 불러오기 완료")
print("=" * 60)
print(f"행 수: {df.shape[0]}")
print(f"열 수: {df.shape[1]}")
print("=" * 60)

# 상위 5개 행 미리보기
print("\n[상위 5개 행 미리보기]")
df.head()
