# ============================================================
# [Full EDA Report — Part 1/3] 초기 설정 + 기초통계 + 분포
# ============================================================
# 전처리된 데이터셋(dataset_traffic_accident_cleaned.csv) 기반으로
# 가능한 모든 탐색적 데이터 분석을 수행하여
# traffic_accident_full_EDA_report.pdf 파일로 저장합니다.
#
# Part 1: 데이터 개요, 기초통계, 타겟 분포, 범주형·수치형 분포
# Part 2: 상관관계, 카이제곱 검정, 변수-타겟 관계
# Part 3: 상호작용, 복합위험, 종합 대시보드, 저장
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ----------------------------------------------------------
# 0) 전역 설정
# ----------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 데이터 불러오기
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident_cleaned.csv'
df = pd.read_csv(data_path)

# 공통 색상
SEV = {0: '#3498db', 1: '#e74c3c'}
SEV_LBL = {0: 'Non-Severe (0)', 1: 'Severe (1)'}
TD_LBL = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
TD_CLR = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}

num_cols = ['Traffic_Density', 'Speed_Limit', 'Driver_Age', 'Driver_Experience', 'Accident_Severity']
cat_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']

# PDF 저장 경로
pdf_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/traffic_accident_full_EDA_report.pdf'

# 헤더 유틸 함수
def page_header(fig, title, sub=''):
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    if sub:
        fig.text(0.5, 0.94, sub, ha='center', fontsize=9, style='italic', color='gray')

# PDF 객체 생성 (Part 3에서 닫음)
pdf = PdfPages(pdf_path)

# ============================================================
# PAGE 1: 데이터 개요 및 기초통계 요약
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
page_header(fig, 'Page 1. Dataset Overview & Descriptive Statistics',
            f'Cleaned Dataset: {df.shape[0]} rows × {df.shape[1]} columns | No missing values')

# (1-1) 데이터 타입 분포
dtype_counts = df.dtypes.map(lambda x: 'Numeric (int64)' if 'int' in str(x) else 'Categorical (object)')
dtype_vc = dtype_counts.value_counts()
axes[0, 0].pie(dtype_vc.values, labels=dtype_vc.index, autopct='%1.0f%%',
               colors=['#3498db', '#e74c3c'], startangle=90, textprops={'fontsize': 11})
axes[0, 0].set_title(f'Variable Types\n({len(df.columns)} total)', fontsize=12, fontweight='bold')

# (1-2) 수치형 변수 기초통계 표
stats_data = []
for col in num_cols:
    stats_data.append([col, f'{df[col].min()}', f'{df[col].mean():.2f}',
                       f'{df[col].median():.1f}', f'{df[col].max()}',
                       f'{df[col].std():.2f}', f'{df[col].skew():.2f}'])
axes[0, 1].axis('off')
table = axes[0, 1].table(
    cellText=stats_data,
    colLabels=['Variable', 'Min', 'Mean', 'Median', 'Max', 'Std', 'Skew'],
    loc='center', cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.4)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f8f9fa')
axes[0, 1].set_title('Numeric Variables Summary', fontsize=12, fontweight='bold', pad=20)

# (1-3) 범주형 변수 고유값 수
cat_unique = [(col, df[col].nunique(), df[col].mode()[0]) for col in cat_cols]
axes[1, 0].axis('off')
table2 = axes[1, 0].table(
    cellText=[(c, str(u), m, f'{df[df[c]==m].shape[0]} ({df[df[c]==m].shape[0]/len(df)*100:.1f}%)')
              for c, u, m in cat_unique],
    colLabels=['Variable', 'Unique', 'Mode', 'Mode Count (%)'],
    loc='center', cellLoc='center'
)
table2.auto_set_font_size(False)
table2.set_fontsize(8.5)
table2.scale(1, 1.4)
for (r, c), cell in table2.get_celld().items():
    if r == 0:
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f8f9fa')
axes[1, 0].set_title('Categorical Variables Summary', fontsize=12, fontweight='bold', pad=20)

# (1-4) 타겟 변수 분포
counts = df['Accident_Severity'].value_counts().sort_index()
bars = axes[1, 1].bar([SEV_LBL[i] for i in counts.index], counts.values,
                       color=[SEV[i] for i in counts.index], edgecolor='black', linewidth=0.5, width=0.5)
for bar, cnt in zip(bars, counts.values):
    axes[1, 1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
                     f'{cnt} ({cnt/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
axes[1, 1].set_title('Target: Accident_Severity', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_ylim(0, max(counts)*1.25)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 1 저장 완료")

# ============================================================
# PAGE 2: 전체 수치형 변수 분포 (히스토그램 + 박스플롯)
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
page_header(fig, 'Page 2. Numeric Variable Distributions',
            'Top: Histogram (Mean=Red, Median=Green) | Bottom: Box Plot by Severity')

for i, col in enumerate(num_cols):
    # 히스토그램
    axes[0, i].hist(df[col], bins=20, color='#5dade2', edgecolor='black', linewidth=0.5, alpha=0.8)
    axes[0, i].axvline(df[col].mean(), color='red', linestyle='--', lw=1.5, label=f'Mean:{df[col].mean():.1f}')
    axes[0, i].axvline(df[col].median(), color='green', linestyle='-.', lw=1.5, label=f'Med:{df[col].median():.1f}')
    axes[0, i].set_title(col, fontsize=10, fontweight='bold')
    axes[0, i].legend(fontsize=6)
    axes[0, i].tick_params(labelsize=7)

    # 박스플롯
    d0 = df[df['Accident_Severity']==0][col]
    d1 = df[df['Accident_Severity']==1][col]
    bp = axes[1, i].boxplot([d0, d1], labels=['Sev=0','Sev=1'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(SEV[0]); bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(SEV[1]); bp['boxes'][1].set_alpha(0.6)
    axes[1, i].set_title(f'{col}\nby Severity', fontsize=9, fontweight='bold')
    axes[1, i].tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 2 저장 완료")

# ============================================================
# PAGE 3: 범주형 변수 빈도 분포 (6개 변수)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
page_header(fig, 'Page 3. Categorical Variable Frequency Distributions')

palette_list = ['#5dade2','#58d68d','#f0b27a','#bb8fce','#f1948a','#85c1e9']
for i, col in enumerate(cat_cols):
    ax = axes[i//3, i%3]
    vc = df[col].value_counts()
    bars = ax.barh(vc.index[::-1], vc.values[::-1], color=palette_list[i],
                    edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vc.values[::-1]):
        ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
                f'{v} ({v/len(df)*100:.1f}%)', va='center', fontsize=9)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('Count')

plt.tight_layout(rect=[0, 0, 1, 0.94])
pdf.savefig(fig); plt.close()
print("✔ Page 3 저장 완료")

# ============================================================
# PAGE 4: 범주형 변수별 심각 사고 비율
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
page_header(fig, 'Page 4. Severe Accident Rate by Categorical Variables',
            'Stacked 100% bars — Red portion = Severe Rate')

for i, col in enumerate(cat_cols):
    ax = axes[i//3, i%3]
    order = df.groupby(col)['Accident_Severity'].mean().sort_values(ascending=False).index
    ct = pd.crosstab(df[col], df['Accident_Severity'], normalize='index') * 100
    ct = ct.loc[order]
    ct.plot(kind='barh', stacked=True, ax=ax, color=[SEV[0], SEV[1]],
            edgecolor='black', linewidth=0.5)
    for j, (idx, row) in enumerate(ct.iterrows()):
        if row[1] > 5:
            ax.text(row[0]+row[1]/2, j, f'{row[1]:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('%')
    ax.set_ylabel('')
    ax.legend(['Non-Severe','Severe'], loc='lower right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.94])
pdf.savefig(fig); plt.close()
print("✔ Page 4 저장 완료")

# ============================================================
# PAGE 5: 수치형 변수별 심각 사고 상세 비교
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
page_header(fig, 'Page 5. Key Numeric Variables vs Severity (Detail)')

# (5-1) Traffic_Density 그룹바
td_ct = pd.crosstab(df['Traffic_Density'], df['Accident_Severity'])
x = range(len(td_ct)); w = 0.35
axes[0,0].bar([i-w/2 for i in x], td_ct[0], w, label='Non-Severe', color=SEV[0], edgecolor='black', lw=0.5)
axes[0,0].bar([i+w/2 for i in x], td_ct[1], w, label='Severe', color=SEV[1], edgecolor='black', lw=0.5)
axes[0,0].set_xticks(list(x)); axes[0,0].set_xticklabels([TD_LBL[i] for i in td_ct.index])
axes[0,0].set_title('Traffic Density — Count by Severity', fontsize=12, fontweight='bold')
axes[0,0].legend(fontsize=9); axes[0,0].set_ylabel('Count')

# (5-2) Traffic_Density 심각비율
td_pct = pd.crosstab(df['Traffic_Density'], df['Accident_Severity'], normalize='index')*100
bars = axes[0,1].bar([TD_LBL[i] for i in td_pct.index], td_pct[1],
                      color=[TD_CLR[i] for i in td_pct.index], edgecolor='black', lw=0.5, width=0.5)
for bar, v in zip(bars, td_pct[1]):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[0,1].set_title('Traffic Density — Severe Rate', fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('Severe Rate (%)'); axes[0,1].set_ylim(0, td_pct[1].max()*1.3)

# (5-3) Speed_Limit 히스토그램
for s in [0, 1]:
    sub = df[df['Accident_Severity']==s]['Speed_Limit']
    axes[1,0].hist(sub, bins=15, alpha=0.6, label=SEV_LBL[s], color=SEV[s], edgecolor='black', lw=0.5)
    axes[1,0].axvline(sub.mean(), color=SEV[s], linestyle='--', lw=2, alpha=0.8)
axes[1,0].set_title('Speed Limit — Distribution by Severity', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Speed Limit'); axes[1,0].set_ylabel('Count'); axes[1,0].legend(fontsize=9)

# (5-4) Speed_Limit 박스플롯
df_tmp = df.copy(); df_tmp['Sev'] = df_tmp['Accident_Severity'].map(SEV_LBL)
sns.boxplot(data=df_tmp, x='Sev', y='Speed_Limit', palette=[SEV[0], SEV[1]], ax=axes[1,1])
for s in [0, 1]:
    m = df[df['Accident_Severity']==s]['Speed_Limit'].mean()
    axes[1,1].text(s, m+2, f'Mean:{m:.1f}', ha='center', fontsize=9, fontweight='bold', color=SEV[s])
axes[1,1].set_title('Speed Limit — Box Plot', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('Speed Limit')

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 5 저장 완료")
print("\n>>> Part 1 완료 — Part 2 셀을 실행하세요 >>>")
