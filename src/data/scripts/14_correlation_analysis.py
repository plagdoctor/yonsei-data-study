# ============================================================
# [Step 14] 변수 간 상관관계 분석
# ============================================================
# 전처리 완료된 데이터셋(dataset_traffic_accident_cleaned.csv)을
# 기반으로 변수 간 상관관계를 분석합니다.
#   - 수치형 변수 간 피어슨 상관계수 행렬
#   - 상관관계 히트맵 시각화
#   - 타겟 변수(Accident_Severity)와의 상관관계 정리
#   - 범주형 변수와 타겟 변수 간 관계 분석 (카이제곱 검정)
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------
# 0) 한글 폰트 설정 (Colab 환경)
# ----------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'  # Colab 기본 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# ----------------------------------------------------------
# 1) 전처리 완료 데이터 불러오기
# ----------------------------------------------------------
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident_cleaned.csv'
df = pd.read_csv(data_path)

print("=" * 70)
print("[Step 14] 변수 간 상관관계 분석")
print("=" * 70)
print(f"데이터: {df.shape[0]}행 × {df.shape[1]}열\n")

# ----------------------------------------------------------
# 2) 수치형 변수 피어슨 상관계수 행렬
# ----------------------------------------------------------
num_cols = ['Traffic_Density', 'Speed_Limit', 'Driver_Age',
            'Driver_Experience', 'Accident_Severity']

corr_matrix = df[num_cols].corr().round(4)  # 피어슨 상관계수 계산

print("=" * 70)
print("[1] 수치형 변수 피어슨 상관계수 행렬")
print("=" * 70)
print(corr_matrix.to_string())

# ----------------------------------------------------------
# 3) 상관계수 해석 기준 출력
# ----------------------------------------------------------
print("\n※ 상관계수 해석 기준:")
print("  |r| < 0.1  → 무시할 수준 (거의 무관)")
print("  0.1 ≤ |r| < 0.3 → 약한 상관")
print("  0.3 ≤ |r| < 0.5 → 보통 상관")
print("  0.5 ≤ |r| < 0.7 → 강한 상관")
print("  |r| ≥ 0.7  → 매우 강한 상관")

# ----------------------------------------------------------
# 4) 타겟 변수(Accident_Severity)와의 상관관계 정리
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[2] Accident_Severity(타겟)와 수치형 변수 간 상관관계")
print("=" * 70)

# 타겟과의 상관계수를 절대값 기준 내림차순 정렬
target_corr = corr_matrix['Accident_Severity'].drop('Accident_Severity')
target_corr_abs = target_corr.abs().sort_values(ascending=False)

# 상관 강도 해석 함수
def interpret_corr(r):
    """상관계수 절대값에 따른 해석 문자열 반환"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "무시할 수준"
    elif abs_r < 0.3:
        return "약한 상관"
    elif abs_r < 0.5:
        return "보통 상관"
    elif abs_r < 0.7:
        return "강한 상관"
    else:
        return "매우 강한 상관"

# 방향 해석 함수
def interpret_direction(r):
    """상관계수 부호에 따른 방향 문자열 반환"""
    if r > 0:
        return "양(+)의 상관"
    elif r < 0:
        return "음(-)의 상관"
    else:
        return "무상관"

# 결과 테이블 생성
target_result = []
for col in target_corr_abs.index:
    r = target_corr[col]
    target_result.append({
        '변수': col,
        '상관계수': round(r, 4),
        '|상관계수|': round(abs(r), 4),
        '방향': interpret_direction(r),
        '강도': interpret_corr(r)
    })

target_df = pd.DataFrame(target_result)
print(target_df.to_string(index=False))

# ----------------------------------------------------------
# 5) 상관관계 히트맵 시각화
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[3] 상관관계 히트맵 시각화")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# 히트맵 생성 (상관계수 값 표시, 색상 맵 적용)
sns.heatmap(
    corr_matrix,
    annot=True,             # 셀에 상관계수 값 표시
    fmt='.4f',              # 소수점 4자리까지 표시
    cmap='RdBu_r',          # 빨강(양) ~ 파랑(음) 색상 맵
    center=0,               # 0을 기준으로 색상 대칭
    vmin=-1, vmax=1,        # 색상 범위 -1 ~ 1
    square=True,            # 정사각형 셀
    linewidths=0.5,         # 셀 구분선 두께
    cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'}
)

ax.set_title('Pearson Correlation Heatmap (Numeric Variables)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

print("✔ 히트맵 출력 완료")

# ----------------------------------------------------------
# 6) 범주형 변수와 타겟 간 관계 (카이제곱 검정)
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[4] 범주형 변수와 Accident_Severity 간 관계 (카이제곱 검정)")
print("=" * 70)

cat_cols = ['Weather', 'Road_Type', 'Time_of_Day',
            'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']

chi2_results = []
for col in cat_cols:
    # 교차표(크로스탭) 생성
    contingency = pd.crosstab(df[col], df['Accident_Severity'])

    # 카이제곱 검정 수행
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # 크래머의 V 계산 (효과 크기)
    n = contingency.sum().sum()              # 전체 관측수
    min_dim = min(contingency.shape) - 1     # min(행수, 열수) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))  # 크래머의 V 공식

    # 유의성 판단 (유의수준 0.05 기준)
    significance = "유의함 (p < 0.05)" if p_value < 0.05 else "유의하지 않음 (p ≥ 0.05)"

    # 효과 크기 해석
    def interpret_cramers_v(v):
        """크래머의 V 값에 따른 효과 크기 해석"""
        if v < 0.1:
            return "약함"
        elif v < 0.3:
            return "보통"
        elif v < 0.5:
            return "강함"
        else:
            return "매우 강함"

    chi2_results.append({
        '변수': col,
        '카이제곱(χ²)': round(chi2, 4),
        'p-value': f"{p_value:.6f}" if p_value >= 0.000001 else f"{p_value:.2e}",
        '자유도': dof,
        "Cramer's V": round(cramers_v, 4),
        '효과 크기': interpret_cramers_v(cramers_v),
        '유의성': significance
    })

chi2_df = pd.DataFrame(chi2_results)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 150)
print(chi2_df.to_string(index=False))

# ----------------------------------------------------------
# 7) 범주형 변수별 사고 심각도 비율표
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("[5] 범주형 변수별 Accident_Severity=1 비율")
print("=" * 70)

for col in cat_cols:
    print(f"\n▶ {col}")
    # 각 범주별 사고 심각도 평균 (= 심각 사고 비율)
    severity_rate = df.groupby(col)['Accident_Severity'].agg(
        ['count', 'sum', 'mean']
    ).round(4)
    severity_rate.columns = ['전체 건수', '심각(1) 건수', '심각 비율']
    severity_rate = severity_rate.sort_values('심각 비율', ascending=False)
    severity_rate['심각 비율(%)'] = (severity_rate['심각 비율'] * 100).round(2)
    print(severity_rate[['전체 건수', '심각(1) 건수', '심각 비율(%)']].to_string())
