# ============================================================
# [Step 11] 전처리 4단계 — 전처리 검증
# ============================================================
# 1~3단계 전처리가 올바르게 적용되었는지 검증합니다.
#   - 결측치 잔존 여부 확인
#   - 이상치 재검사 (Speed_Limit, Driver_Age)
#   - 데이터 타입 확인
#   - 수치형 변수 기초통계 재출력
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

print("=" * 60)
print("[4단계] 전처리 검증")
print("=" * 60)

# ----------------------------------------------------------
# 1) 결측치 잔존 여부 확인
# ----------------------------------------------------------
print("\n[검증 1] 결측치 잔존 여부")
print("-" * 40)
total_missing = df.isnull().sum().sum()  # 전체 결측치 합계
print(f"전체 결측치 수: {total_missing}건")
if total_missing == 0:
    print("✔ 결측치 처리 완료 — 잔존 결측치 없음")
else:
    print("✘ 결측치가 남아 있습니다:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# ----------------------------------------------------------
# 2) 이상치 재검사
# ----------------------------------------------------------
print("\n[검증 2] 이상치 재검사")
print("-" * 40)

# Speed_Limit: IQR 상한 초과 여부 확인
Q1_s = df['Speed_Limit'].quantile(0.25)
Q3_s = df['Speed_Limit'].quantile(0.75)
IQR_s = Q3_s - Q1_s
upper_s = Q3_s + 1.5 * IQR_s
outlier_speed = (df['Speed_Limit'] > upper_s).sum()
print(f"Speed_Limit — 범위: {df['Speed_Limit'].min()} ~ {df['Speed_Limit'].max()}, IQR 이상치: {outlier_speed}건")

# Driver_Age: 18세 미만 여부 확인
under18 = (df['Driver_Age'] < 18).sum()
print(f"Driver_Age — 범위: {df['Driver_Age'].min()} ~ {df['Driver_Age'].max()}, 18세 미만: {under18}건")

if outlier_speed == 0 and under18 == 0:
    print("✔ 이상치 처리 완료 — 잔존 이상치 없음")

# ----------------------------------------------------------
# 3) 데이터 타입 확인
# ----------------------------------------------------------
print("\n[검증 3] 데이터 타입 확인")
print("-" * 40)
print(df.dtypes)

# Traffic_Density, Driver_Experience가 int인지 확인
td_ok = df['Traffic_Density'].dtype in ['int64', 'int32']
de_ok = df['Driver_Experience'].dtype in ['int64', 'int32']
if td_ok and de_ok:
    print("\n✔ 데이터 타입 변환 완료 — Traffic_Density, Driver_Experience 모두 int")

# ----------------------------------------------------------
# 4) 수치형 변수 기초통계 재출력
# ----------------------------------------------------------
print("\n[검증 4] 전처리 후 수치형 변수 기초통계")
print("-" * 40)
print(df.describe().round(2).to_string())
