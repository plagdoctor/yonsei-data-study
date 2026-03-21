# ============================================================
# [Step 17] EDA 시각화 Part 3 — 종합 대시보드
# ============================================================
# 타겟 변수를 중심으로 모든 핵심 변수를 한눈에 비교할 수 있는
# 종합 시각화를 생성합니다.
#   - 전체 변수 중요도 바 차트 (상관계수 + Cramer's V)
#   - 핵심 4개 변수의 심각 사고 조건 종합
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
# 0) 설정
# ----------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ============================================================
# [시각화 7] 전체 변수 — 타겟과의 연관 강도 종합 비교
# ============================================================

# 수치형 변수: 피어슨 상관계수의 절대값
num_associations = {
    'Traffic_Density': abs(df['Traffic_Density'].corr(df['Accident_Severity'])),
    'Speed_Limit': abs(df['Speed_Limit'].corr(df['Accident_Severity'])),
    'Driver_Age': abs(df['Driver_Age'].corr(df['Accident_Severity'])),
    'Driver_Experience': abs(df['Driver_Experience'].corr(df['Accident_Severity']))
}

# 범주형 변수: Cramer's V 계산
cat_cols = ['Weather', 'Road_Type', 'Time_of_Day',
            'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
cat_associations = {}

for col in cat_cols:
    contingency = pd.crosstab(df[col], df['Accident_Severity'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    cat_associations[col] = cramers_v

# 전체 합치기
all_associations = {**num_associations, **cat_associations}
assoc_df = pd.DataFrame({
    'Variable': list(all_associations.keys()),
    'Association': list(all_associations.values()),
    'Type': (['Numeric'] * len(num_associations) +
             ['Categorical'] * len(cat_associations))
}).sort_values('Association', ascending=True)

# 바 차트 생성
fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#e74c3c' if t == 'Categorical' else '#3498db'
          for t in assoc_df['Type']]

bars = ax.barh(
    assoc_df['Variable'], assoc_df['Association'],
    color=colors, edgecolor='black', linewidth=0.5, height=0.6
)

# 각 바 옆에 수치 표시
for bar, val in zip(bars, assoc_df['Association']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

# 유의성 기준선
ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(0.105, len(assoc_df) - 0.5, 'Weak (0.1)', fontsize=9, color='gray')
ax.axvline(x=0.3, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(0.305, len(assoc_df) - 0.5, 'Moderate (0.3)', fontsize=9, color='gray')

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', edgecolor='black', label='Numeric (|Pearson r|)'),
    Patch(facecolor='#e74c3c', edgecolor='black', label="Categorical (Cramer's V)")
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

ax.set_title("Figure 7. Variable Association Strength with Accident Severity",
             fontsize=14, fontweight='bold')
ax.set_xlabel('Association Strength', fontsize=12)
ax.set_xlim(0, 0.7)

plt.tight_layout()
plt.show()

# ============================================================
# [시각화 8] 고위험 조건 조합 — Top 10 심각 사고 비율
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Weather × Road_Condition × Traffic_Density 조합별 심각 비율 계산
td_map = {0: 'Low', 1: 'Medium', 2: 'High'}
df_temp = df.copy()
df_temp['TD_Label'] = df_temp['Traffic_Density'].map(td_map)

combo = df_temp.groupby(['Weather', 'Road_Condition', 'TD_Label']).agg(
    total=('Accident_Severity', 'count'),
    severe=('Accident_Severity', 'sum'),
    severe_rate=('Accident_Severity', 'mean')
).reset_index()

# 최소 10건 이상인 조합만 필터링 (신뢰성 확보)
combo = combo[combo['total'] >= 10]
combo['severe_rate_pct'] = (combo['severe_rate'] * 100).round(1)
combo['label'] = combo['Weather'] + ' + ' + combo['Road_Condition'] + ' + TD:' + combo['TD_Label']

# 심각 비율 상위 10개 조합
top10 = combo.nlargest(10, 'severe_rate_pct').sort_values('severe_rate_pct', ascending=True)

# 색상: 심각도에 따른 그라데이션
norm = plt.Normalize(vmin=0, vmax=100)
cmap = plt.cm.RdYlGn_r
bar_colors = [cmap(norm(val)) for val in top10['severe_rate_pct']]

bars = ax.barh(
    top10['label'], top10['severe_rate_pct'],
    color=bar_colors, edgecolor='black', linewidth=0.5, height=0.6
)

# 각 바 옆에 비율과 건수 표시
for bar, (_, row) in zip(bars, top10.iterrows()):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{row["severe_rate_pct"]}% ({int(row["severe"])}/{int(row["total"])}건)',
            ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_title('Figure 8. Top 10 High-Risk Condition Combinations (Severe Rate)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Severe Accident Rate (%)', fontsize=12)
ax.set_xlim(0, 110)

plt.tight_layout()
plt.show()

# ============================================================
# 분석 결과 텍스트 요약 출력
# ============================================================
print("=" * 70)
print("EDA 시각화 분석 결과 요약")
print("=" * 70)

print("""
[Figure 1] 타겟 변수 분포
  - Non-Severe(0) 71.5% vs Severe(1) 28.5%로 클래스 불균형 존재

[Figure 2] 핵심 범주형 변수
  - Weather: Rainy 66.8% > Stormy 44.4% > Clear 10.4%
  - Road_Condition: Wet 65.6% > Icy 35.6% > Dry 8.8%
  → 비/습한 조건에서 사고 심각도가 급격히 증가

[Figure 3] 핵심 수치형 변수
  - Traffic_Density: 밀도가 높을수록 심각 사고 비율 증가 (r=0.44)
  - Speed_Limit: 제한속도가 높을수록 심각 사고 소폭 증가 (r=0.25)

[Figure 4] Weather × Road_Condition 상호작용
  - Rainy+Wet 조합이 가장 높은 심각 비율 → 두 변수의 강한 연관성 확인

[Figure 5] Traffic_Density × Speed_Limit 상호작용
  - 교통 밀도 High + 제한속도 High 구간에서 심각 비율 극대화

[Figure 6] Driver_Age × Driver_Experience 다중공선성
  - r=0.9453의 매우 강한 선형 관계 → 모델링 시 하나 제거 필요

[Figure 7] 전체 변수 연관 강도 종합
  - Road_Condition(0.57) > Weather(0.55) > Traffic_Density(0.44) 순

[Figure 8] 고위험 조건 조합
  - 가장 위험한 조합: Rainy + Wet + 높은 교통 밀도
""")
