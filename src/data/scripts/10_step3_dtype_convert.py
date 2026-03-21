# ============================================================
# [Step 10] 전처리 3단계 — 데이터 타입 변환
# ============================================================
# 결측치 대체 후 float64로 남아 있는 변수를 int로 변환합니다.
#   - Traffic_Density: float64 → int (0, 1, 2 순서형)
#   - Driver_Experience: float64 → int (운전 경력 년수)
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("[3단계] 데이터 타입 변환")
print("=" * 60)

# ----------------------------------------------------------
# 처리 전 데이터 타입 확인
# ----------------------------------------------------------
print("\n▶ 처리 전 데이터 타입:")
print(f"  - Traffic_Density: {df['Traffic_Density'].dtype}")
print(f"  - Driver_Experience: {df['Driver_Experience'].dtype}")

# ----------------------------------------------------------
# Traffic_Density: float64 → int 변환
# ----------------------------------------------------------
df['Traffic_Density'] = df['Traffic_Density'].astype(int)

# ----------------------------------------------------------
# Driver_Experience: float64 → int 변환
# ----------------------------------------------------------
df['Driver_Experience'] = df['Driver_Experience'].astype(int)

# ----------------------------------------------------------
# 처리 후 데이터 타입 확인
# ----------------------------------------------------------
print("\n▶ 처리 후 데이터 타입:")
print(f"  - Traffic_Density: {df['Traffic_Density'].dtype}")
print(f"  - Driver_Experience: {df['Driver_Experience'].dtype}")
print("\n✔ 데이터 타입 변환 완료")
