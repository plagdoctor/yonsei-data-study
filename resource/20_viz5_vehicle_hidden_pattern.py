# ============================================================
# [시각화 5] 차량 유형별 숨겨진 패턴 — 조건부 스택 바 차트
# ============================================================
# 분석 질문: 차량 유형(Car/Truck/Bus)은 전체적으로는 사고 심각도와
#            유의하지 않지만, 혼잡한 고속도로에서는 대형 차량이
#            더 위험한가?
# 관련 변수: Vehicle_Type, Road_Type, Traffic_Density, Accident_Severity
# 시각화 유형: 조건별 그룹 바 차트 + 전체 vs 특정 조건 비교
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
plt.rcParams['figure.dpi'] = 120

# 전처리 완료 데이터 불러오기
data_path = '/content/drive/MyDrive/대학원/생성형AI를활용한데이터과학실무/3rd-week/data/dataset_traffic_accident_cleaned.csv'
df = pd.read_csv(data_path)

# 색상 정의
vehicle_colors = {'Car': '#3498db', 'Truck': '#e67e22', 'Bus': '#e74c3c'}
vehicle_order = ['Car', 'Truck', 'Bus']

# ----------------------------------------------------------
# 1) 4가지 조건에서 차량 유형별 심각 비율 비교
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

fig.suptitle('Vehicle Type Hidden Patterns — Severe Rate by Conditions',
             fontsize=15, fontweight='bold', y=1.02)

# 분석할 4가지 조건 정의
conditions = [
    {
        'title': 'Overall (All Data)',
        'subtitle': 'n=840',
        'filter': df,
        'ax': axes[0, 0]
    },
    {
        'title': 'Highway Only',
        'subtitle': '',
        'filter': df[df['Road_Type'] == 'Highway'],
        'ax': axes[0, 1]
    },
    {
        'title': 'Highway + High Traffic Density',
        'subtitle': '',
        'filter': df[(df['Road_Type'] == 'Highway') & (df['Traffic_Density'] == 2)],
        'ax': axes[1, 0]
    },
    {
        'title': 'Highway + High Density + Wet/Rainy',
        'subtitle': '',
        'filter': df[(df['Road_Type'] == 'Highway') &
                     (df['Traffic_Density'] == 2) &
                     ((df['Road_Condition'] == 'Wet') | (df['Weather'] == 'Rainy'))],
        'ax': axes[1, 1]
    }
]

for cond in conditions:
    ax = cond['ax']
    df_sub = cond['filter']

    # 차량 유형별 심각 비율 및 건수 계산
    results = []
    for vtype in vehicle_order:
        sub = df_sub[df_sub['Vehicle_Type'] == vtype]
        total = len(sub)
        if total > 0:
            severe = sub['Accident_Severity'].sum()
            rate = severe / total * 100
        else:
            severe = 0
            rate = 0
        results.append({
            'type': vtype,
            'total': total,
            'severe': int(severe),
            'rate': rate
        })

    # 막대 그래프 그리기
    x_pos = range(len(results))
    bars = ax.bar(
        [r['type'] for r in results],
        [r['rate'] for r in results],
        color=[vehicle_colors[r['type']] for r in results],
        edgecolor='black', linewidth=0.5, width=0.55
    )

    # 각 바 위에 비율과 건수 표시
    for bar, r in zip(bars, results):
        if r['total'] > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{r["rate"]:.1f}%\n({r["severe"]}/{r["total"]})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 해당 조건의 전체 심각 비율 기준선
    if len(df_sub) > 0:
        overall = df_sub['Accident_Severity'].mean() * 100
        ax.axhline(y=overall, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(len(results) - 0.5, overall + 1, f'Avg: {overall:.1f}%',
                fontsize=9, color='gray', fontweight='bold', ha='right')

    # 제목 설정
    sub_n = len(df_sub)
    ax.set_title(f'{cond["title"]}\n(n={sub_n})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Severe Accident Rate (%)', fontsize=10)
    ax.set_ylim(0, min(max([r['rate'] for r in results]) * 1.4, 105))

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 2) Road_Type × Vehicle_Type × Traffic_Density 종합 히트맵
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fig.suptitle('Severe Rate (%) by Vehicle Type × Road Type at Each Traffic Density',
             fontsize=14, fontweight='bold', y=1.05)

td_map = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
road_type_order = ['Highway', 'City Road', 'Rural Road', 'Mountain Road']

for idx, td in enumerate([0, 1, 2]):
    df_sub = df[df['Traffic_Density'] == td]

    # 교차 심각 비율 계산
    cross = df_sub.groupby(['Vehicle_Type', 'Road_Type']).agg(
        rate=('Accident_Severity', 'mean'),
        count=('Accident_Severity', 'count')
    ).reset_index()

    # 피벗 테이블
    pivot_rate = cross.pivot(index='Vehicle_Type', columns='Road_Type', values='rate') * 100
    pivot_count = cross.pivot(index='Vehicle_Type', columns='Road_Type', values='count')

    # 순서 정렬
    avail_vtype = [v for v in vehicle_order if v in pivot_rate.index]
    avail_road = [r for r in road_type_order if r in pivot_rate.columns]
    pivot_rate = pivot_rate.reindex(index=avail_vtype, columns=avail_road)
    pivot_count = pivot_count.reindex(index=avail_vtype, columns=avail_road)

    # annot 텍스트
    annot = pivot_rate.copy().astype(str)
    for v in avail_vtype:
        for r in avail_road:
            rv = pivot_rate.loc[v, r] if not pd.isna(pivot_rate.loc[v, r]) else np.nan
            cv = pivot_count.loc[v, r] if not pd.isna(pivot_count.loc[v, r]) else 0
            if pd.isna(rv) or cv == 0:
                annot.loc[v, r] = '-'
            else:
                annot.loc[v, r] = f'{rv:.0f}%\n(n={int(cv)})'

    sns.heatmap(
        pivot_rate, annot=annot, fmt='',
        cmap='YlOrRd', vmin=0, vmax=100,
        linewidths=0.8, linecolor='white',
        ax=axes[idx],
        cbar=True if idx == 2 else False,
        cbar_kws={'label': 'Severe Rate (%)', 'shrink': 0.8} if idx == 2 else {},
        annot_kws={'fontsize': 9}
    )

    sub_rate = df_sub['Accident_Severity'].mean() * 100
    axes[idx].set_title(f'Density: {td_map[td]}\n(Avg Severe: {sub_rate:.1f}%)',
                         fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Road Type', fontsize=10)
    axes[idx].set_ylabel('Vehicle Type' if idx == 0 else '', fontsize=10)
    if idx > 0:
        axes[idx].set_yticklabels([])

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3) 결과 요약 출력
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("차량 유형별 심각 사고 비율 — 조건별 비교")
print("=" * 70)
print(f"\n{'조건':<40} {'Car':<15} {'Truck':<15} {'Bus':<15}")
print("-" * 85)

for cond in conditions:
    df_sub = cond['filter']
    rates = []
    for vtype in vehicle_order:
        sub = df_sub[df_sub['Vehicle_Type'] == vtype]
        if len(sub) >= 5:
            rate = sub['Accident_Severity'].mean() * 100
            rates.append(f'{rate:.1f}% (n={len(sub)})')
        else:
            rates.append(f'N/A (n={len(sub)})')
    print(f'{cond["title"]:<40} {rates[0]:<15} {rates[1]:<15} {rates[2]:<15}')
