# ============================================================
# [Step 15] EDA 시각화 Part 1 — 핵심 변수 분포
# ============================================================
# 상관관계 분석에서 타겟(Accident_Severity)과 유의미한
# 관계가 확인된 핵심 변수들의 분포를 시각화합니다.
#   - 핵심 수치형: Traffic_Density, Speed_Limit
#   - 핵심 범주형: Weather, Road_Condition
#   - 타겟 변수: Accident_Severity (클래스 분포)
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 0) 설정
# ----------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 전처리 완료 데이터 불러오기
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident_cleaned.csv'
df = pd.read_csv(data_path)

# 색상 팔레트 정의
severity_colors = {0: '#3498db', 1: '#e74c3c'}  # 파랑: 비심각, 빨강: 심각
severity_labels = {0: 'Non-Severe (0)', 1: 'Severe (1)'}

# ============================================================
# [시각화 1] 타겟 변수 클래스 분포
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1-1) 타겟 변수 바 차트
counts = df['Accident_Severity'].value_counts().sort_index()
bars = axes[0].bar(
    [severity_labels[i] for i in counts.index],
    counts.values,
    color=[severity_colors[i] for i in counts.index],
    edgecolor='black', linewidth=0.5
)
# 바 위에 건수 및 비율 표시
for bar, count in zip(bars, counts.values):
    pct = count / len(df) * 100
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
axes[0].set_title('Accident Severity Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, max(counts.values) * 1.25)

# (1-2) 타겟 변수 파이 차트
axes[1].pie(
    counts.values,
    labels=[severity_labels[i] for i in counts.index],
    colors=[severity_colors[i] for i in counts.index],
    autopct='%1.1f%%', startangle=90,
    explode=[0, 0.05], shadow=True,
    textprops={'fontsize': 11}
)
axes[1].set_title('Accident Severity Proportion', fontsize=13, fontweight='bold')

plt.suptitle('Figure 1. Target Variable (Accident_Severity) Distribution',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# [시각화 2] 핵심 범주형 변수 — Weather & Road_Condition
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (2-1) Weather별 사고 심각도 비율
weather_order = df.groupby('Weather')['Accident_Severity'].mean().sort_values(ascending=False).index
weather_ct = pd.crosstab(df['Weather'], df['Accident_Severity'], normalize='index') * 100
weather_ct = weather_ct.loc[weather_order]

weather_ct.plot(
    kind='barh', stacked=True, ax=axes[0],
    color=[severity_colors[0], severity_colors[1]],
    edgecolor='black', linewidth=0.5
)
# 각 바에 심각 사고 비율(%) 표시
for i, (idx, row) in enumerate(weather_ct.iterrows()):
    severe_pct = row[1]
    axes[0].text(severe_pct / 2 + row[0], i, f'{severe_pct:.1f}%',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')
axes[0].set_title('Severity Rate by Weather', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Percentage (%)')
axes[0].set_ylabel('')
axes[0].legend(['Non-Severe (0)', 'Severe (1)'], loc='lower right')

# (2-2) Road_Condition별 사고 심각도 비율
road_order = df.groupby('Road_Condition')['Accident_Severity'].mean().sort_values(ascending=False).index
road_ct = pd.crosstab(df['Road_Condition'], df['Accident_Severity'], normalize='index') * 100
road_ct = road_ct.loc[road_order]

road_ct.plot(
    kind='barh', stacked=True, ax=axes[1],
    color=[severity_colors[0], severity_colors[1]],
    edgecolor='black', linewidth=0.5
)
for i, (idx, row) in enumerate(road_ct.iterrows()):
    severe_pct = row[1]
    axes[1].text(severe_pct / 2 + row[0], i, f'{severe_pct:.1f}%',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')
axes[1].set_title('Severity Rate by Road Condition', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Percentage (%)')
axes[1].set_ylabel('')
axes[1].legend(['Non-Severe (0)', 'Severe (1)'], loc='lower right')

plt.suptitle('Figure 2. Key Categorical Variables vs Accident Severity',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# [시각화 3] 핵심 수치형 변수 — Traffic_Density & Speed_Limit
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (3-1) Traffic_Density별 사고 심각도 비율
td_ct = pd.crosstab(df['Traffic_Density'], df['Accident_Severity'])
td_pct = pd.crosstab(df['Traffic_Density'], df['Accident_Severity'], normalize='index') * 100

td_labels = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
x_pos = range(len(td_ct.index))

# 그룹 바 차트
width = 0.35
bars0 = axes[0].bar([x - width/2 for x in x_pos], td_ct[0], width,
                     label='Non-Severe (0)', color=severity_colors[0], edgecolor='black', linewidth=0.5)
bars1 = axes[0].bar([x + width/2 for x in x_pos], td_ct[1], width,
                     label='Severe (1)', color=severity_colors[1], edgecolor='black', linewidth=0.5)

# 각 바 위에 심각 비율 표시
for x, density in zip(x_pos, td_ct.index):
    severe_pct = td_pct.loc[density, 1]
    total = td_ct.loc[density].sum()
    axes[0].text(x, total * 0.55, f'Severe\n{severe_pct:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([td_labels.get(i, str(i)) for i in td_ct.index])
axes[0].set_title('Severity by Traffic Density', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Traffic Density')
axes[0].set_ylabel('Count')
axes[0].legend()

# (3-2) Speed_Limit 분포 — 사고 심각도별
for sev in [0, 1]:
    subset = df[df['Accident_Severity'] == sev]['Speed_Limit']
    axes[1].hist(subset, bins=15, alpha=0.6,
                 label=severity_labels[sev], color=severity_colors[sev],
                 edgecolor='black', linewidth=0.5)
    # 각 그룹의 평균선 표시
    axes[1].axvline(subset.mean(), color=severity_colors[sev],
                    linestyle='--', linewidth=2, alpha=0.8)
    axes[1].text(subset.mean() + 1, axes[1].get_ylim()[1] * (0.9 - sev * 0.1),
                 f'Mean: {subset.mean():.1f}', color=severity_colors[sev],
                 fontsize=10, fontweight='bold')

axes[1].set_title('Speed Limit Distribution by Severity', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Speed Limit')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.suptitle('Figure 3. Key Numeric Variables vs Accident Severity',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("✔ Part 1 시각화 완료 (Figure 1~3)")
