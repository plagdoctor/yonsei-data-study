# ============================================================
# [Step 13] 전처리 전·후 비교 요약
# ============================================================
# 원본 데이터를 다시 불러와 전처리된 데이터와 비교합니다.
# 변경 사항을 항목별로 정리하여 표 형태로 출력합니다.
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# ----------------------------------------------------------
# 1) 원본 데이터 다시 불러오기 (비교용)
# ----------------------------------------------------------
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident.csv'
df_original = pd.read_csv(data_path)  # 원본 데이터
df_cleaned = df.copy()                # 전처리된 현재 데이터

# ----------------------------------------------------------
# 2) 전처리 전·후 결측치 비교
# ----------------------------------------------------------
print("=" * 70)
print("[비교 1] 결측치 변화")
print("=" * 70)

missing_compare = pd.DataFrame({
    '전처리 전': df_original.isnull().sum(),
    '전처리 후': df_cleaned.isnull().sum(),
    '변화': df_original.isnull().sum() - df_cleaned.isnull().sum()
})
# 변화가 있는 변수만 필터링
missing_changed = missing_compare[missing_compare['변화'] > 0]
print(missing_changed.to_string())
print(f"\n총 처리된 결측치: {missing_changed['변화'].sum()}건 → 0건")

# ----------------------------------------------------------
# 3) 전처리 전·후 수치형 기초통계 비교
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[비교 2] 수치형 변수 기초통계 변화")
print("=" * 70)

num_cols = ['Traffic_Density', 'Speed_Limit', 'Driver_Age',
            'Driver_Experience', 'Accident_Severity']

for col in num_cols:
    # 전처리 전·후 값이 달라진 변수만 상세 출력
    before_min = df_original[col].min()
    after_min = df_cleaned[col].min()
    before_max = df_original[col].max()
    after_max = df_cleaned[col].max()
    before_mean = df_original[col].mean()
    after_mean = df_cleaned[col].mean()
    before_std = df_original[col].std()
    after_std = df_cleaned[col].std()

    # 변화 여부 판단 (소수점 2자리 기준)
    changed = (round(before_min, 2) != round(after_min, 2) or
               round(before_max, 2) != round(after_max, 2) or
               round(before_mean, 2) != round(after_mean, 2))

    if changed:
        print(f"\n▶ {col} — 변화 있음")
        compare_stats = pd.DataFrame({
            '항목': ['최소값', '평균', '중위값', '최대값', '표준편차'],
            '전처리 전': [
                round(before_min, 2),
                round(before_mean, 2),
                round(df_original[col].median(), 2),
                round(before_max, 2),
                round(before_std, 2)
            ],
            '전처리 후': [
                round(after_min, 2),
                round(after_mean, 2),
                round(df_cleaned[col].median(), 2),
                round(after_max, 2),
                round(after_std, 2)
            ]
        })
        compare_stats['변화'] = compare_stats['전처리 후'] - compare_stats['전처리 전']
        print(compare_stats.to_string(index=False))
    else:
        print(f"\n▶ {col} — 변화 없음")

# ----------------------------------------------------------
# 4) 전처리 전·후 데이터 타입 비교
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[비교 3] 데이터 타입 변화")
print("=" * 70)

dtype_compare = pd.DataFrame({
    '전처리 전': df_original.dtypes.astype(str),
    '전처리 후': df_cleaned.dtypes.astype(str)
})
# 타입이 변경된 변수만 필터링
dtype_changed = dtype_compare[dtype_compare['전처리 전'] != dtype_compare['전처리 후']]
if len(dtype_changed) > 0:
    print(dtype_changed.to_string())
else:
    print("변경된 데이터 타입 없음")

# ----------------------------------------------------------
# 5) 전체 변경 사항 종합 요약표
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[종합] 전처리 변경 사항 요약표")
print("=" * 70)

summary_data = [
    {
        '단계': '1단계',
        '처리 유형': '결측치 대체',
        '변수': 'Weather',
        '전처리 전': f'결측 {df_original["Weather"].isnull().sum()}건',
        '전처리 후': f'결측 {df_cleaned["Weather"].isnull().sum()}건',
        '처리 내용': f'최빈값({df_original["Weather"].mode()[0]})으로 대체'
    },
    {
        '단계': '1단계',
        '처리 유형': '결측치 대체',
        '변수': 'Traffic_Density',
        '전처리 전': f'결측 {df_original["Traffic_Density"].isnull().sum()}건',
        '전처리 후': f'결측 {df_cleaned["Traffic_Density"].isnull().sum()}건',
        '처리 내용': f'중위값({df_original["Traffic_Density"].median()})으로 대체'
    },
    {
        '단계': '1단계',
        '처리 유형': '결측치 대체',
        '변수': 'Driver_Experience',
        '전처리 전': f'결측 {df_original["Driver_Experience"].isnull().sum()}건',
        '전처리 후': f'결측 {df_cleaned["Driver_Experience"].isnull().sum()}건',
        '처리 내용': f'중위값({df_original["Driver_Experience"].median()})으로 대체'
    },
    {
        '단계': '2단계',
        '처리 유형': '이상치 클리핑',
        '변수': 'Speed_Limit',
        '전처리 전': f'범위: {df_original["Speed_Limit"].min()}~{df_original["Speed_Limit"].max()}',
        '전처리 후': f'범위: {df_cleaned["Speed_Limit"].min()}~{df_cleaned["Speed_Limit"].max()}',
        '처리 내용': 'IQR 상한(110) 초과 → 110으로 클리핑'
    },
    {
        '단계': '2단계',
        '처리 유형': '이상치 클리핑',
        '변수': 'Driver_Age',
        '전처리 전': f'범위: {df_original["Driver_Age"].min()}~{df_original["Driver_Age"].max()}',
        '전처리 후': f'범위: {df_cleaned["Driver_Age"].min()}~{df_cleaned["Driver_Age"].max()}',
        '처리 내용': '18세 미만 → 18로 클리핑'
    },
    {
        '단계': '3단계',
        '처리 유형': '타입 변환',
        '변수': 'Traffic_Density',
        '전처리 전': str(df_original['Traffic_Density'].dtype),
        '전처리 후': str(df_cleaned['Traffic_Density'].dtype),
        '처리 내용': 'float64 → int 변환'
    },
    {
        '단계': '3단계',
        '처리 유형': '타입 변환',
        '변수': 'Driver_Experience',
        '전처리 전': str(df_original['Driver_Experience'].dtype),
        '전처리 후': str(df_cleaned['Driver_Experience'].dtype),
        '처리 내용': 'float64 → int 변환'
    }
]

summary_df = pd.DataFrame(summary_data)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 150)
print(summary_df.to_string(index=False))

# ----------------------------------------------------------
# 6) 전체 데이터 형태 비교
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[최종] 데이터셋 크기 비교")
print("=" * 70)
print(f"전처리 전: {df_original.shape[0]}행 × {df_original.shape[1]}열")
print(f"전처리 후: {df_cleaned.shape[0]}행 × {df_cleaned.shape[1]}열")
print("✔ 행 삭제 없이 전체 840건 유지")
