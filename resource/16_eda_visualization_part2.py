# ============================================================
# [Step 16] EDA 시각화 Part 2 — 변수 간 상호작용
# ============================================================
# 타겟과 유의미한 관계가 있는 핵심 변수들 사이의
# 상호작용(Interaction)을 시각화합니다.
#   - Weather × Road_Condition 교차 분석
#   - Traffic_Density × Speed_Limit 상호작용
#   - Driver_Age × Driver_Experience 다중공선성 확인
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

severity_colors = {0: '#3498db', 1: '#e74c3c'}

# ============================================================
# [시각화 4] Weather × Road_Condition 교차 히트맵
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (4-1) Weather × Road_Condition 교차 빈도
cross_count = pd.crosstab(df['Weather'], df['Road_Condition'])

# 빈도가 높은 순서로 정렬
weather_order = ['Rainy', 'Stormy', 'Snowy', 'Foggy', 'Clear']
road_order = ['Wet', 'Icy', 'Under Construction', 'Dry']
cross_count = cross_count.reindex(index=weather_order, columns=road_order)

sns.heatmap(
    cross_count, annot=True, fmt='d', cmap='YlOrRd',
    linewidths=0.5, ax=axes[0],
    cbar_kws={'label': 'Count'}
)
axes[0].set_title('Weather × Road Condition (Frequency)',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel('Road Condition')
axes[0].set_ylabel('Weather')

# (4-2) Weather × Road_Condition별 심각 사고 비율 히트맵
cross_severity = df.groupby(['Weather', 'Road_Condition'])['Accident_Severity'].mean() * 100
cross_severity = cross_severity.unstack()
cross_severity = cross_severity.reindex(index=weather_order, columns=road_order)

sns.heatmap(
    cross_severity, annot=True, fmt='.1f', cmap='RdYlGn_r',
    linewidths=0.5, ax=axes[1],
    vmin=0, vmax=100,
    cbar_kws={'label': 'Severe Rate (%)'}
)
axes[1].set_title('Weather × Road Condition (Severe Rate %)',
                   fontsize=13, fontweight='bold')
axes[1].set_xlabel('Road Condition')
axes[1].set_ylabel('Weather')

plt.suptitle('Figure 4. Interaction: Weather × Road Condition',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# [시각화 5] Traffic_Density × Speed_Limit 상호작용
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

td_labels = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
td_colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

# (5-1) Traffic_Density별 Speed_Limit 박스플롯 (심각도 구분)
# 심각도별로 분리하여 Traffic_Density × Speed_Limit 패턴 확인
df_plot = df.copy()
df_plot['TD_Label'] = df_plot['Traffic_Density'].map(td_labels)
df_plot['Severity_Label'] = df_plot['Accident_Severity'].map(
    {0: 'Non-Severe (0)', 1: 'Severe (1)'}
)

sns.boxplot(
    data=df_plot,
    x='TD_Label', y='Speed_Limit', hue='Severity_Label',
    palette=[severity_colors[0], severity_colors[1]],
    ax=axes[0], order=['Low (0)', 'Medium (1)', 'High (2)']
)
axes[0].set_title('Speed Limit by Traffic Density & Severity',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel('Traffic Density')
axes[0].set_ylabel('Speed Limit')
axes[0].legend(title='Severity')

# (5-2) Traffic_Density × Speed_Limit 구간별 심각 사고 비율
# Speed_Limit을 3개 구간으로 분할
df_plot['Speed_Group'] = pd.cut(
    df_plot['Speed_Limit'],
    bins=[0, 50, 80, 110],
    labels=['Low (≤50)', 'Medium (51-80)', 'High (81-110)']
)

# 교차표 생성: Traffic_Density × Speed_Group별 심각 비율
interaction_rate = df_plot.groupby(
    ['Traffic_Density', 'Speed_Group']
)['Accident_Severity'].mean() * 100

interaction_pivot = interaction_rate.unstack(level=0)
interaction_pivot.columns = [td_labels.get(c, c) for c in interaction_pivot.columns]

interaction_pivot.plot(
    kind='bar', ax=axes[1],
    color=[td_colors[0], td_colors[1], td_colors[2]],
    edgecolor='black', linewidth=0.5
)
# 각 바 위에 비율 표시
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)

axes[1].set_title('Severe Rate by Speed Group × Traffic Density',
                   fontsize=13, fontweight='bold')
axes[1].set_xlabel('Speed Limit Group')
axes[1].set_ylabel('Severe Accident Rate (%)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='Traffic Density')
axes[1].set_ylim(0, 100)

plt.suptitle('Figure 5. Interaction: Traffic Density × Speed Limit',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================
# [시각화 6] Driver_Age × Driver_Experience — 다중공선성 확인
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# 산점도: 심각도별 색상 구분
for sev in [0, 1]:
    subset = df[df['Accident_Severity'] == sev]
    ax.scatter(
        subset['Driver_Age'], subset['Driver_Experience'],
        c=severity_colors[sev],
        label=f'{"Severe (1)" if sev == 1 else "Non-Severe (0)"}',
        alpha=0.4, s=30, edgecolors='white', linewidth=0.3
    )

# 회귀선 추가
z = np.polyfit(df['Driver_Age'], df['Driver_Experience'], 1)  # 1차 회귀
p = np.poly1d(z)
age_range = np.linspace(df['Driver_Age'].min(), df['Driver_Age'].max(), 100)
ax.plot(age_range, p(age_range), '--', color='black', linewidth=2, alpha=0.7,
        label=f'Regression (r = {df["Driver_Age"].corr(df["Driver_Experience"]):.4f})')

ax.set_title('Figure 6. Driver Age vs Experience (Multicollinearity Check)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Driver Age', fontsize=12)
ax.set_ylabel('Driver Experience (years)', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()

print("✔ Part 2 시각화 완료 (Figure 4~6)")
