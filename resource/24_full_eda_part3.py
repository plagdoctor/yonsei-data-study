# ============================================================
# [Full EDA Report — Part 3/3] 차량유형 + 복합위험 + Top10 + 요약
# ============================================================
# Part 1·2에서 생성한 pdf 객체에 이어서 페이지를 추가하고,
# 최종적으로 pdf.close()로 파일을 완성합니다.
# ============================================================

# ============================================================
# PAGE 11: 차량 유형별 숨겨진 패턴 (조건별 비교)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
page_header(fig, 'Page 11. Vehicle Type — Hidden Patterns by Conditions',
            'Severe Rate comparison across increasingly restrictive conditions')

# 차량 유형 순서 및 색상
vehicle_order = ['Car', 'Truck', 'Bus']
vehicle_colors = {'Car': '#3498db', 'Truck': '#e67e22', 'Bus': '#e74c3c'}

# 4가지 조건 정의
conditions = [
    {'title': 'Overall (All Data)', 'data': df},
    {'title': 'Highway Only', 'data': df[df['Road_Type'] == 'Highway']},
    {'title': 'Highway + High Density',
     'data': df[(df['Road_Type'] == 'Highway') & (df['Traffic_Density'] == 2)]},
    {'title': 'Highway + High Density\n+ Wet/Rainy',
     'data': df[(df['Road_Type'] == 'Highway') & (df['Traffic_Density'] == 2) &
               ((df['Road_Condition'] == 'Wet') | (df['Weather'] == 'Rainy'))]}
]

for ci, cond in enumerate(conditions):
    ax = axes[ci // 2, ci % 2]
    df_sub = cond['data']
    results = []
    for vtype in vehicle_order:
        sub = df_sub[df_sub['Vehicle_Type'] == vtype]
        total = len(sub)
        severe = int(sub['Accident_Severity'].sum()) if total > 0 else 0
        rate = severe / total * 100 if total > 0 else 0
        results.append({'type': vtype, 'total': total, 'severe': severe, 'rate': rate})

    bars = ax.bar([r['type'] for r in results], [r['rate'] for r in results],
                  color=[vehicle_colors[r['type']] for r in results],
                  edgecolor='black', linewidth=0.5, width=0.55)
    for bar, r in zip(bars, results):
        if r['total'] > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{r["rate"]:.1f}%\n({r["severe"]}/{r["total"]})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    if len(df_sub) > 0:
        overall = df_sub['Accident_Severity'].mean() * 100
        ax.axhline(y=overall, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(2.4, overall + 1, f'Avg: {overall:.1f}%', fontsize=8, color='gray', fontweight='bold')
    ax.set_title(f'{cond["title"]}\n(n={len(df_sub)})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Severe Rate (%)')
    ax.set_ylim(0, min(max([r['rate'] for r in results]) * 1.4 + 5, 105))

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 11 저장 완료")

# ============================================================
# PAGE 12: 복합 위험 점수 분석 (4-패널 대시보드)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
page_header(fig, 'Page 12. Compound Risk Score Analysis',
            'Risk Factors: Bad Weather + Bad Road + High Density + High Speed (>80)')

# 위험 요인 플래그 생성
df['Risk_Weather'] = (df['Weather'] != 'Clear').astype(int)
df['Risk_Road'] = (df['Road_Condition'] != 'Dry').astype(int)
df['Risk_Density'] = (df['Traffic_Density'] == 2).astype(int)
df['Risk_Speed'] = (df['Speed_Limit'] > 80).astype(int)
df['Risk_Score'] = df['Risk_Weather'] + df['Risk_Road'] + df['Risk_Density'] + df['Risk_Speed']

# 위험 점수별 통계
risk_stats = df.groupby('Risk_Score').agg(
    total=('Accident_Severity', 'count'),
    severe=('Accident_Severity', 'sum'),
    rate=('Accident_Severity', 'mean')
).reset_index()
risk_stats['rate_pct'] = (risk_stats['rate'] * 100).round(1)
risk_stats['pct_of_total'] = (risk_stats['total'] / len(df) * 100).round(1)

# (12-1) 위험 점수별 심각 사고 비율
colors_gradient = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
bars = axes[0, 0].bar(risk_stats['Risk_Score'], risk_stats['rate_pct'],
                       color=[colors_gradient[i] for i in risk_stats['Risk_Score']],
                       edgecolor='black', linewidth=0.8, width=0.6)
for bar, (_, row) in zip(bars, risk_stats.iterrows()):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{row["rate_pct"]}%\n({int(row["severe"])}/{int(row["total"])})',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
avg_rate = df['Accident_Severity'].mean() * 100
axes[0, 0].axhline(y=avg_rate, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 0].text(4.3, avg_rate + 1, f'Avg: {avg_rate:.1f}%', fontsize=8, color='gray', fontweight='bold')
axes[0, 0].axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 0].set_title('Severe Rate by Risk Score', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Number of Risk Factors (0~4)')
axes[0, 0].set_ylabel('Severe Accident Rate (%)')
axes[0, 0].set_xticks(range(5))
axes[0, 0].set_xticklabels(['0\n(Safe)', '1', '2', '3', '4\n(Most\nDangerous)'])
axes[0, 0].set_ylim(0, max(risk_stats['rate_pct']) * 1.25)

# (12-2) 건수 분포
bars2 = axes[0, 1].bar(risk_stats['Risk_Score'], risk_stats['total'],
                        color=[colors_gradient[i] for i in risk_stats['Risk_Score']],
                        edgecolor='black', linewidth=0.8, width=0.6, alpha=0.8)
for bar, (_, row) in zip(bars2, risk_stats.iterrows()):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{int(row["total"])}\n({row["pct_of_total"]}%)',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0, 1].set_title('Data Distribution by Risk Score', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Risk Factors (0~4)')
axes[0, 1].set_ylabel('Number of Accidents')
axes[0, 1].set_xticks(range(5))
axes[0, 1].set_ylim(0, max(risk_stats['total']) * 1.25)

# (12-3) 개별 위험 요인 기여도
risk_factors = {
    'Bad Weather\n(Not Clear)': 'Risk_Weather',
    'Bad Road\n(Not Dry)': 'Risk_Road',
    'High Density\n(TD=2)': 'Risk_Density',
    'High Speed\n(>80 km/h)': 'Risk_Speed'
}
factor_rates = []
for label, col in risk_factors.items():
    exposed = df[df[col] == 1]['Accident_Severity'].mean() * 100
    not_exposed = df[df[col] == 0]['Accident_Severity'].mean() * 100
    factor_rates.append({'factor': label, 'exposed': exposed,
                         'not_exposed': not_exposed, 'diff': exposed - not_exposed})
factor_df = pd.DataFrame(factor_rates).sort_values('diff', ascending=True)

y_pos = range(len(factor_df))
axes[1, 0].barh([y - 0.2 for y in y_pos], factor_df['not_exposed'], height=0.35,
                 label='Without Factor', color='#3498db', edgecolor='black', lw=0.5, alpha=0.7)
axes[1, 0].barh([y + 0.2 for y in y_pos], factor_df['exposed'], height=0.35,
                 label='With Factor', color='#e74c3c', edgecolor='black', lw=0.5, alpha=0.7)
for y, (_, row) in zip(y_pos, factor_df.iterrows()):
    axes[1, 0].text(row['exposed'] + 1, y + 0.2, f'+{row["diff"]:.1f}%p',
                     va='center', fontsize=9, fontweight='bold', color='#c0392b')
axes[1, 0].set_yticks(list(y_pos))
axes[1, 0].set_yticklabels(factor_df['factor'])
axes[1, 0].set_title('Individual Risk Factor Impact', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Severe Accident Rate (%)')
axes[1, 0].legend(loc='lower right', fontsize=9)

# (12-4) 위험 점수별 요인 구성 비율 (스택 바)
risk_composition = df.groupby('Risk_Score')[
    ['Risk_Weather', 'Risk_Road', 'Risk_Density', 'Risk_Speed']
].mean() * 100
factor_labels = ['Bad Weather', 'Bad Road', 'High Density', 'High Speed']
factor_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bottom = np.zeros(len(risk_composition))
for i, (col, label) in enumerate(zip(
    ['Risk_Weather', 'Risk_Road', 'Risk_Density', 'Risk_Speed'], factor_labels
)):
    vals = risk_composition[col].values
    axes[1, 1].bar(risk_composition.index, vals, bottom=bottom,
                    label=label, color=factor_colors[i],
                    edgecolor='white', linewidth=0.5, width=0.6)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 15:
            axes[1, 1].text(risk_composition.index[j], b + v/2, f'{v:.0f}%',
                             ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    bottom += vals
axes[1, 1].set_title('Risk Factor Composition by Score', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Risk Score')
axes[1, 1].set_ylabel('Factor Activation Rate (%)')
axes[1, 1].set_xticks(range(5))
axes[1, 1].legend(loc='upper left', fontsize=8, ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 12 저장 완료")

# ============================================================
# PAGE 13: 고위험 조합 Top 10
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
page_header(fig, 'Page 13. Top 10 High-Risk Combinations',
            'Highest Severe Rate combinations (min 5 observations)')

# 모든 범주형+Traffic_Density 조합에 대한 심각 비율 계산
combo_cols = ['Weather', 'Road_Condition', 'Traffic_Density']
combo = df.groupby(combo_cols).agg(
    total=('Accident_Severity', 'count'),
    severe=('Accident_Severity', 'sum'),
    rate=('Accident_Severity', 'mean')
).reset_index()
combo['rate_pct'] = combo['rate'] * 100
combo['TD_label'] = combo['Traffic_Density'].map(TD_LBL)
combo['label'] = combo['Weather'] + ' + ' + combo['Road_Condition'] + '\n(TD=' + combo['TD_label'] + ')'

# 최소 5건 이상인 조합만 필터링 후 상위 10개
combo_top = combo[combo['total'] >= 5].sort_values('rate_pct', ascending=False).head(10)
combo_top = combo_top.sort_values('rate_pct', ascending=True)  # 수평 바 차트용 역순

# (13-1) 수평 바 차트
cmap_risk = plt.cm.RdYlGn_r(np.linspace(0.3, 1.0, len(combo_top)))
bars = axes[0].barh(range(len(combo_top)), combo_top['rate_pct'],
                     color=cmap_risk, edgecolor='black', linewidth=0.5, height=0.6)
for i, (bar, (_, row)) in enumerate(zip(bars, combo_top.iterrows())):
    axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                  f'{row["rate_pct"]:.1f}% ({int(row["severe"])}/{int(row["total"])})',
                  ha='left', va='center', fontsize=9, fontweight='bold')
axes[0].set_yticks(range(len(combo_top)))
axes[0].set_yticklabels(combo_top['label'], fontsize=8)
axes[0].set_title('Top 10 Highest Severe-Rate Combinations', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Severe Accident Rate (%)')
axes[0].axvline(avg_rate, color='gray', linestyle='--', lw=1.5, alpha=0.7)
axes[0].text(avg_rate + 1, 0, f'Avg: {avg_rate:.1f}%', fontsize=8, color='gray')
axes[0].set_xlim(0, 110)

# (13-2) 상세 표
combo_bottom = combo[combo['total'] >= 5].sort_values('rate_pct', ascending=False).head(10)
tbl_data = []
for rank, (_, row) in enumerate(combo_bottom.iterrows(), 1):
    tbl_data.append([
        f'#{rank}', row['Weather'], row['Road_Condition'],
        row['TD_label'], f'{int(row["total"])}',
        f'{int(row["severe"])}', f'{row["rate_pct"]:.1f}%'
    ])

axes[1].axis('off')
tbl = axes[1].table(
    cellText=tbl_data,
    colLabels=['Rank', 'Weather', 'Road Cond.', 'Density', 'Total', 'Severe', 'Rate'],
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#34495e'); cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f8f9fa')
    # 상위 3개 강조
    if r > 0 and r <= 3:
        cell.set_facecolor('#fde8e8')
    if r > 0 and c == 6:
        cell.set_text_props(fontweight='bold', color='#c0392b')
axes[1].set_title('Detailed Breakdown', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 13 저장 완료")

# ============================================================
# PAGE 14: Driver Age × Experience 상세 분석
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
page_header(fig, 'Page 14. Driver Age & Experience — Detailed Analysis',
            'Age groups, Experience groups, and their interaction with Severity')

# (14-1) Driver_Age 그룹별 심각 비율
age_bins = [17, 25, 35, 45, 55, 65, 80]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '66+']
df['Age_Group'] = pd.cut(df['Driver_Age'], bins=age_bins, labels=age_labels)
age_rate = df.groupby('Age_Group')['Accident_Severity'].agg(['mean', 'count']).reset_index()
age_rate['pct'] = age_rate['mean'] * 100

bars_age = axes[0, 0].bar(age_rate['Age_Group'].astype(str), age_rate['pct'],
                            color='#5dade2', edgecolor='black', lw=0.5, width=0.6)
for bar, (_, row) in zip(bars_age, age_rate.iterrows()):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{row["pct"]:.1f}%\n(n={int(row["count"])})',
                     ha='center', fontsize=9, fontweight='bold')
axes[0, 0].axhline(avg_rate, color='gray', ls='--', lw=1.5, alpha=0.7)
axes[0, 0].set_title('Severe Rate by Age Group', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age Group'); axes[0, 0].set_ylabel('Severe Rate (%)')
axes[0, 0].set_ylim(0, age_rate['pct'].max() * 1.3)

# (14-2) Driver_Experience 그룹별 심각 비율
exp_bins = [-1, 5, 10, 20, 30, 50]
exp_labels = ['0-5', '6-10', '11-20', '21-30', '31+']
df['Exp_Group'] = pd.cut(df['Driver_Experience'], bins=exp_bins, labels=exp_labels)
exp_rate = df.groupby('Exp_Group')['Accident_Severity'].agg(['mean', 'count']).reset_index()
exp_rate['pct'] = exp_rate['mean'] * 100

bars_exp = axes[0, 1].bar(exp_rate['Exp_Group'].astype(str), exp_rate['pct'],
                            color='#58d68d', edgecolor='black', lw=0.5, width=0.6)
for bar, (_, row) in zip(bars_exp, exp_rate.iterrows()):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{row["pct"]:.1f}%\n(n={int(row["count"])})',
                     ha='center', fontsize=9, fontweight='bold')
axes[0, 1].axhline(avg_rate, color='gray', ls='--', lw=1.5, alpha=0.7)
axes[0, 1].set_title('Severe Rate by Experience Group', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Experience (years)'); axes[0, 1].set_ylabel('Severe Rate (%)')
axes[0, 1].set_ylim(0, exp_rate['pct'].max() * 1.3)

# (14-3) Age Group × Density 히트맵
age_density = (df.groupby(['Age_Group', 'Traffic_Density'])['Accident_Severity'].mean() * 100).unstack()
age_density.columns = [TD_LBL[c] for c in age_density.columns]
sns.heatmap(age_density, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
            ax=axes[1, 0], vmin=0, vmax=80, cbar_kws={'label': 'Severe %', 'shrink': 0.8})
axes[1, 0].set_title('Severe Rate: Age Group × Density', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Traffic Density'); axes[1, 0].set_ylabel('Age Group')

# (14-4) Experience Group × Density 히트맵
exp_density = (df.groupby(['Exp_Group', 'Traffic_Density'])['Accident_Severity'].mean() * 100).unstack()
exp_density.columns = [TD_LBL[c] for c in exp_density.columns]
sns.heatmap(exp_density, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
            ax=axes[1, 1], vmin=0, vmax=80, cbar_kws={'label': 'Severe %', 'shrink': 0.8})
axes[1, 1].set_title('Severe Rate: Exp Group × Density', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Traffic Density'); axes[1, 1].set_ylabel('Experience Group')

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 14 저장 완료")

# ============================================================
# PAGE 15: Road_Type × Time_of_Day 상호작용
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
page_header(fig, 'Page 15. Interaction: Road Type × Time of Day',
            'Frequency (left) & Severe Rate % (right)')

road_order = ['Highway', 'City Road', 'Rural Road', 'Mountain Road']
time_order = ['Morning', 'Afternoon', 'Evening', 'Night']

# (15-1) 빈도 히트맵
ct_rt = pd.crosstab(df['Road_Type'], df['Time_of_Day'])
ct_rt = ct_rt.reindex(index=[r for r in road_order if r in ct_rt.index],
                       columns=[t for t in time_order if t in ct_rt.columns])
sns.heatmap(ct_rt, annot=True, fmt='d', cmap='Blues', linewidths=0.5, ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Cross-tabulation (Frequency)', fontsize=12, fontweight='bold')

# (15-2) 심각 비율 히트맵
sr_rt = (df.groupby(['Road_Type', 'Time_of_Day'])['Accident_Severity'].mean() * 100).unstack()
sr_rt = sr_rt.reindex(index=[r for r in road_order if r in sr_rt.index],
                       columns=[t for t in time_order if t in sr_rt.columns])
sns.heatmap(sr_rt, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.5, ax=axes[1],
            vmin=0, vmax=80, cbar_kws={'label': 'Severe Rate (%)'})
axes[1].set_title('Severe Accident Rate (%)', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 15 저장 완료")

# ============================================================
# PAGE 16: Road_Light_Condition 분석
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
page_header(fig, 'Page 16. Road Light Condition Analysis',
            'Distribution and Severe Rate by Light Condition')

# (16-1) 조명 조건별 분포 (Severity별 그룹 바)
light_ct = pd.crosstab(df['Road_Light_Condition'], df['Accident_Severity'])
light_order = light_ct.index.tolist()
x = range(len(light_ct)); w = 0.35
axes[0].bar([i - w/2 for i in x], light_ct[0], w, label='Non-Severe',
             color=SEV[0], edgecolor='black', lw=0.5)
axes[0].bar([i + w/2 for i in x], light_ct[1], w, label='Severe',
             color=SEV[1], edgecolor='black', lw=0.5)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(light_order, rotation=15, ha='right')
axes[0].set_title('Count by Light Condition & Severity', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9); axes[0].set_ylabel('Count')

# (16-2) 조명 조건별 심각 비율
light_rate = df.groupby('Road_Light_Condition')['Accident_Severity'].agg(['mean', 'count']).reset_index()
light_rate['pct'] = light_rate['mean'] * 100
light_rate = light_rate.sort_values('pct', ascending=True)

bars_lr = axes[1].barh(light_rate['Road_Light_Condition'], light_rate['pct'],
                         color='#f0b27a', edgecolor='black', lw=0.5, height=0.5)
for bar, (_, row) in zip(bars_lr, light_rate.iterrows()):
    axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                  f'{row["pct"]:.1f}% (n={int(row["count"])})',
                  ha='left', va='center', fontsize=10, fontweight='bold')
axes[1].axvline(avg_rate, color='gray', ls='--', lw=1.5, alpha=0.7)
axes[1].set_title('Severe Rate by Light Condition', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Severe Rate (%)'); axes[1].set_xlim(0, light_rate['pct'].max() * 1.4)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 16 저장 완료")

# ============================================================
# PAGE 17: 종합 요약 대시보드
# ============================================================
fig = plt.figure(figsize=(16, 12))
page_header(fig, 'Page 17. EDA Summary Dashboard',
            'Key Findings from the Exploratory Data Analysis')

# 그리드 레이아웃
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35,
                        top=0.90, bottom=0.05, left=0.08, right=0.95)

# --- (17-1) 데이터셋 개요 텍스트 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
overview_text = (
    f"Dataset Overview\n"
    f"{'='*40}\n"
    f"Total Records: {len(df)}\n"
    f"Variables: {df.shape[1]} ({len(num_cols)} numeric, {len(cat_cols)} categorical)\n"
    f"Target: Accident_Severity (Binary)\n"
    f"  - Non-Severe (0): {(df['Accident_Severity']==0).sum()} "
    f"({(df['Accident_Severity']==0).mean()*100:.1f}%)\n"
    f"  - Severe (1): {(df['Accident_Severity']==1).sum()} "
    f"({(df['Accident_Severity']==1).mean()*100:.1f}%)\n"
    f"Missing Values: 0 (cleaned)\n"
    f"Outliers: Treated (Speed_Limit, Driver_Age)"
)
ax1.text(0.05, 0.95, overview_text, transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eaf2f8', alpha=0.8))
ax1.set_title('Dataset Overview', fontsize=12, fontweight='bold')

# --- (17-2) 주요 발견 사항 ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

# 상위 4개 변수와 타겟 연관 강도 계산
num_assoc_sum = {c: abs(df[c].corr(df['Accident_Severity'])) for c in num_cols if c != 'Accident_Severity'}
cat_assoc_sum = {}
for col in cat_cols:
    ct = pd.crosstab(df[col], df['Accident_Severity'])
    chi2, pv, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum(); md = min(ct.shape) - 1
    cat_assoc_sum[col] = np.sqrt(chi2 / (n * md))
all_assoc = {**num_assoc_sum, **cat_assoc_sum}
top4 = sorted(all_assoc.items(), key=lambda x: x[1], reverse=True)[:4]

findings_text = (
    f"Key Findings\n"
    f"{'='*40}\n"
    f"[Top 4 Variables by Association]\n"
    f"  1. {top4[0][0]}: {top4[0][1]:.4f}\n"
    f"  2. {top4[1][0]}: {top4[1][1]:.4f}\n"
    f"  3. {top4[2][0]}: {top4[2][1]:.4f}\n"
    f"  4. {top4[3][0]}: {top4[3][1]:.4f}\n\n"
    f"[Multicollinearity]\n"
    f"  Driver_Age x Experience: r={df['Driver_Age'].corr(df['Driver_Experience']):.4f}\n\n"
    f"[Class Imbalance]\n"
    f"  Severe ratio = {df['Accident_Severity'].mean()*100:.1f}%"
)
ax2.text(0.05, 0.95, findings_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e7', alpha=0.8))
ax2.set_title('Key Findings', fontsize=12, fontweight='bold')

# --- (17-3) 상위 5 위험 조합 차트 ---
ax3 = fig.add_subplot(gs[1, :])
combo_top5 = combo[combo['total'] >= 5].sort_values('rate_pct', ascending=False).head(5)
combo_top5 = combo_top5.sort_values('rate_pct', ascending=True)
combo_top5['short_label'] = (combo_top5['Weather'] + ' + ' +
                              combo_top5['Road_Condition'] + ' (TD=' +
                              combo_top5['Traffic_Density'].astype(str) + ')')

cmap5 = plt.cm.Reds(np.linspace(0.4, 0.9, len(combo_top5)))
bars5 = ax3.barh(range(len(combo_top5)), combo_top5['rate_pct'],
                   color=cmap5, edgecolor='black', lw=0.5, height=0.5)
for bar, (_, row) in zip(bars5, combo_top5.iterrows()):
    ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
              f'{row["rate_pct"]:.1f}% (n={int(row["total"])})',
              ha='left', va='center', fontsize=10, fontweight='bold')
ax3.set_yticks(range(len(combo_top5)))
ax3.set_yticklabels(combo_top5['short_label'], fontsize=10)
ax3.set_title('Top 5 Most Dangerous Combinations', fontsize=12, fontweight='bold')
ax3.set_xlabel('Severe Accident Rate (%)')
ax3.axvline(avg_rate, color='gray', ls='--', lw=1.5, alpha=0.7)
ax3.set_xlim(0, 110)

# --- (17-4) 위험 점수 요약 차트 ---
ax4 = fig.add_subplot(gs[2, 0])
risk_bars = ax4.bar(risk_stats['Risk_Score'], risk_stats['rate_pct'],
                     color=[colors_gradient[i] for i in risk_stats['Risk_Score']],
                     edgecolor='black', lw=0.8, width=0.6)
for bar, (_, row) in zip(risk_bars, risk_stats.iterrows()):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
              f'{row["rate_pct"]}%', ha='center', fontsize=9, fontweight='bold')
ax4.set_title('Risk Score Summary', fontsize=12, fontweight='bold')
ax4.set_xlabel('Risk Score (0-4)'); ax4.set_ylabel('Severe Rate (%)')
ax4.set_xticks(range(5))
ax4.set_ylim(0, max(risk_stats['rate_pct']) * 1.2)

# --- (17-5) 분석 결론 텍스트 ---
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

# 임계점 계산
score_2plus = df[df['Risk_Score'] >= 2]
score_3plus = df[df['Risk_Score'] >= 3]

conclusion_text = (
    f"Conclusions & Recommendations\n"
    f"{'='*40}\n"
    f"1. Road Condition & Weather are the\n"
    f"   strongest predictors of severity.\n\n"
    f"2. Traffic Density amplifies all other\n"
    f"   risk factors significantly.\n\n"
    f"3. Risk Score Thresholds:\n"
    f"   - 2+ factors: {score_2plus['Accident_Severity'].mean()*100:.1f}% severe\n"
    f"   - 3+ factors: {score_3plus['Accident_Severity'].mean()*100:.1f}% severe\n\n"
    f"4. Vehicle Type shows hidden patterns\n"
    f"   under specific conditions (Highway\n"
    f"   + High Density).\n\n"
    f"5. Driver Age & Experience have high\n"
    f"   multicollinearity (r>0.94).\n"
    f"   Consider dropping one variable."
)
ax5.text(0.05, 0.95, conclusion_text, transform=ax5.transAxes,
         fontsize=9.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eafaf1', alpha=0.8))
ax5.set_title('Conclusions', fontsize=12, fontweight='bold')

pdf.savefig(fig); plt.close()
print("✔ Page 17 저장 완료")

# ============================================================
# PDF 파일 닫기 — 최종 저장
# ============================================================
pdf.close()
print("\n" + "=" * 60)
print("✅ Full EDA Report 저장 완료!")
print(f"📄 파일 위치: {pdf_path}")
print(f"📊 총 페이지 수: 17페이지")
print("=" * 60)
print("\n[페이지 구성]")
print("  Part 1 (Pages 1-5):  데이터 개요, 기초통계, 분포")
print("  Part 2 (Pages 6-10): 상관관계, 카이제곱, 상호작용")
print("  Part 3 (Pages 11-17): 차량유형, 복합위험, Top10, 운전자분석, 요약")

# 임시 컬럼 정리 (선택)
temp_cols = ['Risk_Weather', 'Risk_Road', 'Risk_Density', 'Risk_Speed',
             'Risk_Score', 'Age_Group', 'Exp_Group']
df.drop(columns=[c for c in temp_cols if c in df.columns], inplace=True)
print("\n🧹 분석용 임시 컬럼 정리 완료")
