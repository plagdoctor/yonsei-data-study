# ============================================================
# [Step 5] 수치형 변수 기초통계 분석 및 표 정리
# ============================================================
# 수치형 변수에 대해 최소값, 평균, 중위값, 최대값, 표준편차,
# 왜곡도(Skewness), 결측치 수, IQR 기반 이상치 수를
# 하나의 표로 정리합니다.
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats  # 왜곡도 계산을 위한 라이브러리

# ----------------------------------------------------------
# 1) 수치형 변수 선택
# ----------------------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ----------------------------------------------------------
# 2) 각 수치형 변수별 기초통계 계산
# ----------------------------------------------------------
# 결과를 저장할 빈 리스트
result_list = []

for col in num_cols:
    # 기초통계량 계산
    col_min = df[col].min()                          # 최소값
    col_mean = df[col].mean()                        # 평균
    col_median = df[col].median()                    # 중위값
    col_max = df[col].max()                          # 최대값
    col_std = df[col].std()                          # 표준편차
    col_skew = df[col].skew()                        # 왜곡도 (Skewness)
    col_missing = df[col].isnull().sum()             # 결측치 수

    # IQR 기반 이상치 수 계산
    Q1 = df[col].quantile(0.25)                      # 1사분위수
    Q3 = df[col].quantile(0.75)                      # 3사분위수
    IQR = Q3 - Q1                                    # 사분위 범위
    lower_bound = Q1 - 1.5 * IQR                     # 이상치 하한
    upper_bound = Q3 + 1.5 * IQR                     # 이상치 상한
    col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]  # 이상치 수

    # 결과 리스트에 추가
    result_list.append({
        '변수명': col,
        '최소값': round(col_min, 2),
        '평균': round(col_mean, 2),
        '중위값': round(col_median, 2),
        '최대값': round(col_max, 2),
        '표준편차': round(col_std, 2),
        '왜곡도': round(col_skew, 2),
        '결측치': col_missing,
        '이상치': col_outliers
    })

# ----------------------------------------------------------
# 3) 표(DataFrame) 형태로 정리하여 출력
# ----------------------------------------------------------
stats_df = pd.DataFrame(result_list)
stats_df = stats_df.set_index('변수명')  # 변수명을 인덱스로 설정

print("=" * 80)
print("수치형 변수 기초통계 요약표")
print("=" * 80)
print(stats_df.to_string())

# ----------------------------------------------------------
# 4) 왜곡도 해석 기준 출력
# ----------------------------------------------------------
print("\n" + "=" * 80)
print("왜곡도(Skewness) 해석 기준")
print("=" * 80)
print("  |왜곡도| < 0.5  → 대체로 대칭 분포")
print("  0.5 ≤ |왜곡도| < 1.0 → 약간 비대칭")
print("  |왜곡도| ≥ 1.0  → 강한 비대칭 (치우침이 큼)")

# ----------------------------------------------------------
# 5) 변수별 해석 출력
# ----------------------------------------------------------
print("\n" + "=" * 80)
print("변수별 분석 결과 해석")
print("=" * 80)

for _, row in stats_df.iterrows():
    name = row.name  # 변수명 (인덱스)
    print(f"\n▶ {name}")

    # 평균과 중위값 비교 → 분포 치우침 방향 판단
    if row['평균'] > row['중위값']:
        direction = "오른쪽으로 치우침 (양의 왜곡)"
    elif row['평균'] < row['중위값']:
        direction = "왼쪽으로 치우침 (음의 왜곡)"
    else:
        direction = "대칭에 가까움"

    # 왜곡도 크기에 따른 비대칭 정도 판단
    abs_skew = abs(row['왜곡도'])
    if abs_skew < 0.5:
        skew_level = "대체로 대칭"
    elif abs_skew < 1.0:
        skew_level = "약간 비대칭"
    else:
        skew_level = "강한 비대칭"

    print(f"  - 범위: {row['최소값']} ~ {row['최대값']}, 평균: {row['평균']}, 중위값: {row['중위값']}")
    print(f"  - 분포: {direction} | 왜곡도 {row['왜곡도']} → {skew_level}")
    print(f"  - 결측치: {int(row['결측치'])}건, 이상치(IQR): {int(row['이상치'])}건")
