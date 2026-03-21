# ============================================================
# [Step 1] Google Drive 마운트
# ============================================================
# Google Colab 환경에서 Google Drive에 저장된 데이터를
# 불러오기 위해 Drive를 마운트합니다.
# 실행 시 Google 계정 인증 팝업이 나타나며,
# 인증 완료 후 /content/drive/MyDrive 경로로 접근할 수 있습니다.
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# Google Drive 마운트 라이브러리 임포트
from google.colab import drive

# Google Drive를 /content/drive 경로에 마운트
drive.mount('/content/drive')

# 마운트 완료 확인 메시지 출력
print("Google Drive 마운트가 완료되었습니다.")
print("데이터 경로: /content/drive/MyDrive/Data/")
