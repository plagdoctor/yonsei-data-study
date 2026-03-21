# ============================================================
# [Step 18] 포괄적 EDA 시각화 세트 — 단일 PDF 저장
# ============================================================
# 전처리된 데이터셋의 주요 분포, 패턴, 관계, 특성을
# 8페이지 시각화로 구성하여 하나의 PDF 파일로 저장합니다.
#
#   Page 1: 타겟 변수(Accident_Severity) 분포
#   Page 2: 핵심 범주형 변수 — Weather & Road_Condition
#   Page 3: 핵심 수치형 변수 — Traffic_Density & Speed_Limit
#   Page 4: 전체 수치형 변수 분포 (히스토그램 + 박스플롯)
#   Page 5: Weather × Road_Condition 상호작용 히트맵
#   Page 6: Traffic_Density × Speed_Limit 상호작용
#   Page 7: 다중공선성 확인 & 상관계수 히트맵
#   Page 8: 변수 중요도 종합 & 고위험 조합 Top 10
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages  # PDF 저장용
from matplotlib.patches import Patch

# ----------------------------------------------------------
# 0) 설정
# ----------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 전처리 완료 데이터 불러오기
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident_cleaned.csv'
df = pd.read_csv(data_path)

# 공통 색상 팔레트
SEV_COLORS = {0: '#3498db', 1: '#e74c3c'}    # 파랑: 비심각, 빨강: 심각
SEV_LABELS = {0: 'Non-Severe (0)', 1: 'Severe (1)'}
TD_LABELS = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
TD_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

# PDF 저장 경로
pdf_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/traffic_accident_visualization.pdf'

# ----------------------------------------------------------
# 공통 유틸 함수
# ----------------------------------------------------------
def add_page_header(fig, title, subtitle=''):
    """각 페이지 상단에 제목과 부제 추가"""
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha='center', fontsize=10,
                 style='italic', color='gray')

# ============================================================
# PDF 생성 시작
# ============================================================
with PdfPages(pdf_path) as pdf:

    # ========================================================
    # PAGE 1: 타겟 변수 분포
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    add_page_header(fig, 'Page 1. Target Variable Distribution',
                    'Accident_Severity: Binary Classification (0 = Non-Severe, 1 = Severe)')

    # (1-1) 바 차트
    counts = df['Accident_Severity'].value_counts().sort_index()
    bars = axes[0].bar(
        [SEV_LABELS[i] for i in counts.index], counts.values,
        color=[SEV_COLORS[i] for i in counts.index],
        edgecolor='black', linewidth=0.5, width=0.5
    )
    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
    axes[0].set_title('Frequency Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_ylim(0, max(counts.values) * 1.25)

    # (1-2) 파이 차트
    axes[1].pie(
        counts.values,
        labels=[SEV_LABELS[i] for i in counts.index],
        colors=[SEV_COLORS[i] for i in counts.index],
        autopct='%1.1f%%', startangle=90, explode=[0, 0.05],
        shadow=True, textprops={'fontsize': 12}
    )
    axes[1].set_title('Class Proportion', fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 1 저장 완료")

    # ========================================================
    # PAGE 2: 핵심 범주형 변수 — Weather & Road_Condition
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    add_page_header(fig, 'Page 2. Key Categorical Variables vs Severity',
                    "Weather (Cramer's V=0.55) & Road_Condition (Cramer's V=0.57)")

    # (2-1) Weather 빈도
    w_order = df.groupby('Weather')['Accident_Severity'].mean().sort_values(ascending=False).index
    w_counts = df['Weather'].value_counts().loc[w_order]
    axes[0, 0].barh(w_counts.index, w_counts.values, color='#5dade2',
                     edgecolor='black', linewidth=0.5)
    for i, v in enumerate(w_counts.values):
        axes[0, 0].text(v + 5, i, str(v), va='center', fontsize=10)
    axes[0, 0].set_title('Weather — Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()

    # (2-2) Weather별 심각 비율
    w_ct = pd.crosstab(df['Weather'], df['Accident_Severity'], normalize='index') * 100
    w_ct = w_ct.loc[w_order]
    w_ct.plot(kind='barh', stacked=True, ax=axes[0, 1],
              color=[SEV_COLORS[0], SEV_COLORS[1]], edgecolor='black', linewidth=0.5)
    for i, (idx, row) in enumerate(w_ct.iterrows()):
        axes[0, 1].text(row[0] + row[1]/2, i, f'{row[1]:.1f}%',
                         ha='center', va='center', fontsize=10,
                         fontweight='bold', color='white')
    axes[0, 1].set_title('Weather — Severe Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Percentage (%)')
    axes[0, 1].set_ylabel('')
    axes[0, 1].legend(['Non-Severe', 'Severe'], loc='lower right', fontsize=9)
    axes[0, 1].invert_yaxis()

    # (2-3) Road_Condition 빈도
    r_order = df.groupby('Road_Condition')['Accident_Severity'].mean().sort_values(ascending=False).index
    r_counts = df['Road_Condition'].value_counts().loc[r_order]
    axes[1, 0].barh(r_counts.index, r_counts.values, color='#58d68d',
                     edgecolor='black', linewidth=0.5)
    for i, v in enumerate(r_counts.values):
        axes[1, 0].text(v + 5, i, str(v), va='center', fontsize=10)
    axes[1, 0].set_title('Road Condition — Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()

    # (2-4) Road_Condition별 심각 비율
    r_ct = pd.crosstab(df['Road_Condition'], df['Accident_Severity'], normalize='index') * 100
    r_ct = r_ct.loc[r_order]
    r_ct.plot(kind='barh', stacked=True, ax=axes[1, 1],
              color=[SEV_COLORS[0], SEV_COLORS[1]], edgecolor='black', linewidth=0.5)
    for i, (idx, row) in enumerate(r_ct.iterrows()):
        axes[1, 1].text(row[0] + row[1]/2, i, f'{row[1]:.1f}%',
                         ha='center', va='center', fontsize=10,
                         fontweight='bold', color='white')
    axes[1, 1].set_title('Road Condition — Severe Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Percentage (%)')
    axes[1, 1].set_ylabel('')
    axes[1, 1].legend(['Non-Severe', 'Severe'], loc='lower right', fontsize=9)
    axes[1, 1].invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 2 저장 완료")

    # ========================================================
    # PAGE 3: 핵심 수치형 변수 — Traffic_Density & Speed_Limit
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    add_page_header(fig, 'Page 3. Key Numeric Variables vs Severity',
                    'Traffic_Density (r=0.44) & Speed_Limit (r=0.25)')

    # (3-1) Traffic_Density별 심각도 그룹 바
    td_ct = pd.crosstab(df['Traffic_Density'], df['Accident_Severity'])
    x_pos = range(len(td_ct.index))
    width = 0.35
    axes[0, 0].bar([x - width/2 for x in x_pos], td_ct[0], width,
                    label='Non-Severe', color=SEV_COLORS[0], edgecolor='black', linewidth=0.5)
    axes[0, 0].bar([x + width/2 for x in x_pos], td_ct[1], width,
                    label='Severe', color=SEV_COLORS[1], edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([TD_LABELS[i] for i in td_ct.index])
    axes[0, 0].set_title('Traffic Density — Count by Severity', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend(fontsize=9)

    # (3-2) Traffic_Density별 심각 비율
    td_pct = pd.crosstab(df['Traffic_Density'], df['Accident_Severity'], normalize='index') * 100
    bars = axes[0, 1].bar(
        [TD_LABELS[i] for i in td_pct.index], td_pct[1],
        color=[TD_COLORS[i] for i in td_pct.index],
        edgecolor='black', linewidth=0.5, width=0.5
    )
    for bar, val in zip(bars, td_pct[1]):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Traffic Density — Severe Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Severe Rate (%)')
    axes[0, 1].set_ylim(0, td_pct[1].max() * 1.3)

    # (3-3) Speed_Limit 히스토그램 (심각도별)
    for sev in [0, 1]:
        subset = df[df['Accident_Severity'] == sev]['Speed_Limit']
        axes[1, 0].hist(subset, bins=15, alpha=0.6, label=SEV_LABELS[sev],
                         color=SEV_COLORS[sev], edgecolor='black', linewidth=0.5)
        axes[1, 0].axvline(subset.mean(), color=SEV_COLORS[sev],
                            linestyle='--', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Speed Limit — Distribution by Severity', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Speed Limit')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend(fontsize=9)

    # (3-4) Speed_Limit 박스플롯 (심각도별)
    df_plot_temp = df.copy()
    df_plot_temp['Severity_Label'] = df_plot_temp['Accident_Severity'].map(SEV_LABELS)
    sns.boxplot(data=df_plot_temp, x='Severity_Label', y='Speed_Limit',
                palette=[SEV_COLORS[0], SEV_COLORS[1]], ax=axes[1, 1])
    # 각 그룹의 평균 표시
    for sev in [0, 1]:
        mean_val = df[df['Accident_Severity'] == sev]['Speed_Limit'].mean()
        axes[1, 1].text(sev, mean_val + 2, f'Mean: {mean_val:.1f}',
                         ha='center', fontsize=10, fontweight='bold', color=SEV_COLORS[sev])
    axes[1, 1].set_title('Speed Limit — Box Plot by Severity', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_ylabel('Speed Limit')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 3 저장 완료")

    # ========================================================
    # PAGE 4: 전체 수치형 변수 분포 (히스토그램 + 박스플롯)
    # ========================================================
    num_cols = ['Traffic_Density', 'Speed_Limit', 'Driver_Age',
                'Driver_Experience', 'Accident_Severity']

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    add_page_header(fig, 'Page 4. All Numeric Variables — Distribution Overview',
                    'Top: Histogram | Bottom: Box Plot (by Severity)')

    for i, col in enumerate(num_cols):
        # 상단: 히스토그램
        axes[0, i].hist(df[col], bins=20, color='#5dade2', edgecolor='black',
                         linewidth=0.5, alpha=0.8)
        axes[0, i].axvline(df[col].mean(), color='red', linestyle='--',
                            linewidth=1.5, label=f'Mean: {df[col].mean():.1f}')
        axes[0, i].axvline(df[col].median(), color='green', linestyle='-.',
                            linewidth=1.5, label=f'Med: {df[col].median():.1f}')
        axes[0, i].set_title(col, fontsize=10, fontweight='bold')
        axes[0, i].legend(fontsize=7)
        axes[0, i].tick_params(labelsize=8)

        # 하단: 박스플롯 (심각도별)
        data_0 = df[df['Accident_Severity'] == 0][col]
        data_1 = df[df['Accident_Severity'] == 1][col]
        bp = axes[1, i].boxplot([data_0, data_1], labels=['Sev=0', 'Sev=1'],
                                 patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(SEV_COLORS[0])
        bp['boxes'][1].set_facecolor(SEV_COLORS[1])
        for box in bp['boxes']:
            box.set_alpha(0.6)
        axes[1, i].set_title(f'{col}\nby Severity', fontsize=9, fontweight='bold')
        axes[1, i].tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 4 저장 완료")

    # ========================================================
    # PAGE 5: Weather × Road_Condition 상호작용
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    add_page_header(fig, 'Page 5. Interaction: Weather × Road Condition',
                    'Frequency (left) & Severe Accident Rate (right)')

    weather_order = ['Rainy', 'Stormy', 'Snowy', 'Foggy', 'Clear']
    road_order = ['Wet', 'Icy', 'Under Construction', 'Dry']

    # (5-1) 교차 빈도 히트맵
    cross_count = pd.crosstab(df['Weather'], df['Road_Condition'])
    cross_count = cross_count.reindex(index=weather_order, columns=road_order)
    sns.heatmap(cross_count, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Cross-tabulation (Frequency)', fontsize=12, fontweight='bold')

    # (5-2) 교차 심각 비율 히트맵
    cross_sev = df.groupby(['Weather', 'Road_Condition'])['Accident_Severity'].mean() * 100
    cross_sev = cross_sev.unstack().reindex(index=weather_order, columns=road_order)
    sns.heatmap(cross_sev, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, ax=axes[1], vmin=0, vmax=100,
                cbar_kws={'label': 'Severe Rate (%)'})
    axes[1].set_title('Severe Accident Rate (%)', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 5 저장 완료")

    # ========================================================
    # PAGE 6: Traffic_Density × Speed_Limit 상호작용
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    add_page_header(fig, 'Page 6. Interaction: Traffic Density × Speed Limit',
                    'Box Plot by Severity (left) & Severe Rate by Speed Group (right)')

    # (6-1) Traffic_Density별 Speed_Limit 박스플롯 (심각도 구분)
    df_p = df.copy()
    df_p['TD_Label'] = df_p['Traffic_Density'].map(TD_LABELS)
    df_p['Sev_Label'] = df_p['Accident_Severity'].map(SEV_LABELS)
    sns.boxplot(data=df_p, x='TD_Label', y='Speed_Limit', hue='Sev_Label',
                palette=[SEV_COLORS[0], SEV_COLORS[1]], ax=axes[0],
                order=['Low (0)', 'Medium (1)', 'High (2)'])
    axes[0].set_title('Speed Limit by Density & Severity', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Traffic Density')
    axes[0].legend(title='Severity', fontsize=9)

    # (6-2) Speed 구간 × Traffic_Density별 심각 비율
    df_p['Speed_Group'] = pd.cut(df_p['Speed_Limit'], bins=[0, 50, 80, 110],
                                  labels=['Low (<=50)', 'Mid (51-80)', 'High (81-110)'])
    interact_rate = df_p.groupby(['Speed_Group', 'Traffic_Density'])['Accident_Severity'].mean() * 100
    interact_pivot = interact_rate.unstack()
    interact_pivot.columns = [TD_LABELS[c] for c in interact_pivot.columns]
    interact_pivot.plot(kind='bar', ax=axes[1],
                        color=[TD_COLORS[0], TD_COLORS[1], TD_COLORS[2]],
                        edgecolor='black', linewidth=0.5)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', fontsize=8, padding=2)
    axes[1].set_title('Severe Rate: Speed Group × Density', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Speed Limit Group')
    axes[1].set_ylabel('Severe Rate (%)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title='Traffic Density', fontsize=9)
    axes[1].set_ylim(0, 100)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 6 저장 완료")

    # ========================================================
    # PAGE 7: 다중공선성 & 상관계수 히트맵
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    add_page_header(fig, 'Page 7. Correlation Analysis',
                    'Multicollinearity Check (left) & Pearson Correlation Heatmap (right)')

    # (7-1) Driver_Age vs Driver_Experience 산점도
    for sev in [0, 1]:
        subset = df[df['Accident_Severity'] == sev]
        axes[0].scatter(subset['Driver_Age'], subset['Driver_Experience'],
                         c=SEV_COLORS[sev], label=SEV_LABELS[sev],
                         alpha=0.35, s=25, edgecolors='white', linewidth=0.3)
    # 회귀선
    z = np.polyfit(df['Driver_Age'], df['Driver_Experience'], 1)
    p_line = np.poly1d(z)
    age_range = np.linspace(df['Driver_Age'].min(), df['Driver_Age'].max(), 100)
    r_val = df['Driver_Age'].corr(df['Driver_Experience'])
    axes[0].plot(age_range, p_line(age_range), '--', color='black', linewidth=2,
                  alpha=0.7, label=f'r = {r_val:.4f}')
    axes[0].set_title('Driver Age vs Experience\n(Multicollinearity)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Driver Age')
    axes[0].set_ylabel('Driver Experience (years)')
    axes[0].legend(fontsize=9)

    # (7-2) 피어슨 상관계수 히트맵
    corr_cols = ['Traffic_Density', 'Speed_Limit', 'Driver_Age',
                 'Driver_Experience', 'Accident_Severity']
    corr_matrix = df[corr_cols].corr().round(4)
    sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                ax=axes[1], cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    axes[1].set_title('Pearson Correlation Heatmap', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 7 저장 완료")

    # ========================================================
    # PAGE 8: 변수 중요도 종합 & 고위험 조합 Top 10
    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    add_page_header(fig, 'Page 8. Variable Importance & High-Risk Combinations',
                    "Association with Severity (left) & Top 10 Risk Combos (right)")

    # (8-1) 전체 변수 연관 강도
    # 수치형: Pearson r 절대값
    num_assoc = {
        'Traffic_Density': abs(df['Traffic_Density'].corr(df['Accident_Severity'])),
        'Speed_Limit': abs(df['Speed_Limit'].corr(df['Accident_Severity'])),
        'Driver_Age': abs(df['Driver_Age'].corr(df['Accident_Severity'])),
        'Driver_Experience': abs(df['Driver_Experience'].corr(df['Accident_Severity']))
    }
    # 범주형: Cramer's V
    cat_cols = ['Weather', 'Road_Type', 'Time_of_Day',
                'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
    cat_assoc = {}
    for col in cat_cols:
        ct = pd.crosstab(df[col], df['Accident_Severity'])
        chi2, _, _, _ = stats.chi2_contingency(ct)
        n = ct.sum().sum()
        min_d = min(ct.shape) - 1
        cat_assoc[col] = np.sqrt(chi2 / (n * min_d))

    all_assoc = {**num_assoc, **cat_assoc}
    assoc_df = pd.DataFrame({
        'Variable': list(all_assoc.keys()),
        'Association': list(all_assoc.values()),
        'Type': (['Numeric'] * 4 + ['Categorical'] * 6)
    }).sort_values('Association', ascending=True)

    bar_colors = ['#e74c3c' if t == 'Categorical' else '#3498db' for t in assoc_df['Type']]
    bars = axes[0].barh(assoc_df['Variable'], assoc_df['Association'],
                         color=bar_colors, edgecolor='black', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, assoc_df['Association']):
        axes[0].text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    axes[0].axvline(0.1, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    axes[0].axvline(0.3, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    axes[0].text(0.105, len(assoc_df) - 0.5, '0.1', fontsize=8, color='gray')
    axes[0].text(0.305, len(assoc_df) - 0.5, '0.3', fontsize=8, color='gray')
    legend_el = [Patch(facecolor='#3498db', edgecolor='black', label='Numeric (|r|)'),
                 Patch(facecolor='#e74c3c', edgecolor='black', label="Categorical (Cramer's V)")]
    axes[0].legend(handles=legend_el, loc='lower right', fontsize=9)
    axes[0].set_title('Variable Association Strength', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Association Strength')
    axes[0].set_xlim(0, 0.7)

    # (8-2) 고위험 조합 Top 10
    td_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    df_t = df.copy()
    df_t['TD_Label'] = df_t['Traffic_Density'].map(td_map)
    combo = df_t.groupby(['Weather', 'Road_Condition', 'TD_Label']).agg(
        total=('Accident_Severity', 'count'),
        severe=('Accident_Severity', 'sum'),
        severe_rate=('Accident_Severity', 'mean')
    ).reset_index()
    combo = combo[combo['total'] >= 10]  # 최소 10건 이상 필터링
    combo['pct'] = (combo['severe_rate'] * 100).round(1)
    combo['label'] = combo['Weather'] + ' + ' + combo['Road_Condition'] + '\n(TD: ' + combo['TD_Label'] + ')'
    top10 = combo.nlargest(10, 'pct').sort_values('pct', ascending=True)

    norm = plt.Normalize(vmin=0, vmax=100)
    cmap = plt.cm.RdYlGn_r
    t_bars = axes[1].barh(top10['label'], top10['pct'],
                           color=[cmap(norm(v)) for v in top10['pct']],
                           edgecolor='black', linewidth=0.5, height=0.6)
    for bar, (_, row) in zip(t_bars, top10.iterrows()):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{row["pct"]}% ({int(row["severe"])}/{int(row["total"])})',
                     ha='left', va='center', fontsize=9, fontweight='bold')
    axes[1].set_title('Top 10 High-Risk Combinations', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Severe Rate (%)')
    axes[1].set_xlim(0, 110)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close()
    print("✔ Page 8 저장 완료")

# ============================================================
# 완료 메시지
# ============================================================
print("\n" + "=" * 70)
print("PDF 저장 완료!")
print("=" * 70)
print(f"파일 경로: {pdf_path}")
print(f"총 페이지: 8페이지")
print("=" * 70)
