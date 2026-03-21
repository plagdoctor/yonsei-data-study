# ============================================================
# [Step 9] 전처리 2단계 — 이상치 처리
# ============================================================
# 이상치가 확인된 2개 변수에 대해 클리핑(Capping)을 적용합니다.
#   - Speed_Limit → IQR 상한(110) 초과 값을 110으로 클리핑
#   - Driver_Age → 18세 미만 값을 18로 클리핑 (도메인 기준)
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("[2단계] 이상치 처리 (클리핑)")
print("=" * 60)

# ----------------------------------------------------------
# Speed_Limit: IQR 기반 상한 클리핑
# ----------------------------------------------------------
# IQR 계산
Q1_speed = df['Speed_Limit'].quantile(0.25)           # 1사분위수
Q3_speed = df['Speed_Limit'].quantile(0.75)           # 3사분위수
IQR_speed = Q3_speed - Q1_speed                       # 사분위 범위
upper_speed = Q3_speed + 1.5 * IQR_speed              # 이상치 상한

# 처리 전 이상치 수 확인
outlier_count_before = (df['Speed_Limit'] > upper_speed).sum()
print(f"\n▶ Speed_Limit")
print(f"  - IQR 상한: {upper_speed}")
print(f"  - 처리 전 이상치 수: {outlier_count_before}건")
print(f"  - 처리 전 최대값: {df['Speed_Limit'].max()}")

# 상한 초과 값을 상한값으로 클리핑
df['Speed_Limit'] = df['Speed_Limit'].clip(upper=upper_speed)

print(f"  - 처리 후 최대값: {df['Speed_Limit'].max()}")
print(f"  ✔ {outlier_count_before}건을 {upper_speed}으로 클리핑 완료")

# ----------------------------------------------------------
# Driver_Age: 도메인 기준 하한 클리핑 (18세 미만 → 18세)
# ----------------------------------------------------------
# 처리 전 18세 미만 수 확인
under18_count = (df['Driver_Age'] < 18).sum()
print(f"\n▶ Driver_Age")
print(f"  - 도메인 하한: 18세 (운전면허 취득 최소 연령)")
print(f"  - 처리 전 18세 미만: {under18_count}건")
print(f"  - 처리 전 최솟값: {df['Driver_Age'].min()}")

# 18세 미만 값을 18로 클리핑
df['Driver_Age'] = df['Driver_Age'].clip(lower=18)

print(f"  - 처리 후 최솟값: {df['Driver_Age'].min()}")
print(f"  ✔ {under18_count}건을 18로 클리핑 완료")
