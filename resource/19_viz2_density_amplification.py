# ============================================================
# [시각화 2] 교통 밀도의 증폭 효과 — 패싯 히트맵
# ============================================================
# 분석 질문: 교통 혼잡도(Traffic_Density)는 다른 위험 요인의
#            효과를 얼마나 증폭시키는가?
# 관련 변수: Traffic_Density, Weather, Road_Condition, Accident_Severity
# 시각화 유형: 패싯 히트맵 (Traffic_Density 수준별 3개 히트맵)
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

# Traffic_Density 라벨 매핑
td_map = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
td_levels = [0, 1, 2]

# 정렬 순서 정의 (심각 비율 높은 순)
weather_order = ['Rainy', 'Stormy', 'Snowy', 'Foggy', 'Clear']
road_order = ['Wet', 'Icy', 'Under Construction', 'Dry']

# ----------------------------------------------------------
# 1) 패싯 히트맵: Traffic_Density 수준별 심각 비율
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

fig.suptitle('Traffic Density Amplification Effect\nSevere Accident Rate (%) by Weather × Road Condition at Each Density Level',
             fontsize=14, fontweight='bold', y=1.05)

# 전체 데이터의 심각 비율 (비교 기준)
overall_rate = df['Accident_Severity'].mean() * 100

for idx, td in enumerate(td_levels):
    # 해당 Traffic_Density 수준의 데이터만 필터링
    df_sub = df[df['Traffic_Density'] == td]

    # Weather × Road_Condition 교차 심각 비율 계산
    cross_rate = df_sub.groupby(['Weather', 'Road_Condition'])['Accident_Severity'].mean() * 100
    cross_rate = cross_rate.unstack()

    # 순서 재정렬 (존재하는 컬럼/인덱스만)
    avail_weather = [w for w in weather_order if w in cross_rate.index]
    avail_road = [r for r in road_order if r in cross_rate.columns]
    cross_rate = cross_rate.reindex(index=avail_weather, columns=avail_road)

    # 건수 행렬도 계산 (셀 내 표기용)
    cross_count = df_sub.groupby(['Weather', 'Road_Condition'])['Accident_Severity'].count()
    cross_count = cross_count.unstack().reindex(index=avail_weather, columns=avail_road)

    # annot 텍스트: 비율% + (건수) 형태
    annot_text = cross_rate.copy().astype(str)
    for w in avail_weather:
        for r in avail_road:
            rate_val = cross_rate.loc[w, r] if not pd.isna(cross_rate.loc[w, r]) else np.nan
            count_val = cross_count.loc[w, r] if not pd.isna(cross_count.loc[w, r]) else 0
            if pd.isna(rate_val) or count_val == 0:
                annot_text.loc[w, r] = '-'
            else:
                annot_text.loc[w, r] = f'{rate_val:.0f}%\n(n={int(count_val)})'

    # 히트맵 그리기
    sns.heatmap(
        cross_rate, annot=annot_text, fmt='',
        cmap='RdYlGn_r', vmin=0, vmax=100,
        linewidths=0.8, linecolor='white',
        ax=axes[idx],
        cbar=True if idx == 2 else False,  # 마지막 축에만 컬러바
        cbar_kws={'label': 'Severe Rate (%)', 'shrink': 0.8} if idx == 2 else {},
        annot_kws={'fontsize': 9}
    )

    # 해당 밀도 수준의 전체 심각 비율
    sub_rate = df_sub['Accident_Severity'].mean() * 100
    sub_count = len(df_sub)

    axes[idx].set_title(
        f'Traffic Density: {td_map[td]}\n(n={sub_count}, Overall Severe Rate: {sub_rate:.1f}%)',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[idx].set_xlabel('Road Condition', fontsize=10)
    axes[idx].set_ylabel('Weather' if idx == 0 else '', fontsize=10)

    # Y축 라벨 첫 번째만 표시
    if idx > 0:
        axes[idx].set_yticklabels([])

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 2) 증폭 효과 요약표 출력
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("교통 밀도별 심각 사고 비율 증폭 효과 요약")
print("=" * 70)

# 주요 조합별 밀도에 따른 심각 비율 변화
key_combos = [
    ('Rainy', 'Wet'),
    ('Clear', 'Dry'),
    ('Snowy', 'Icy'),
    ('Clear', 'Wet')
]

print(f"\n{'날씨':<10} {'도로상태':<20} {'Low(0)':<12} {'Medium(1)':<12} {'High(2)':<12} {'증폭배수':<10}")
print("-" * 76)

for weather, road in key_combos:
    rates = []
    for td in td_levels:
        sub = df[(df['Traffic_Density'] == td) &
                 (df['Weather'] == weather) &
                 (df['Road_Condition'] == road)]
        if len(sub) >= 3:  # 최소 3건 이상
            rate = sub['Accident_Severity'].mean() * 100
            rates.append(f'{rate:.1f}%')
        else:
            rates.append('N/A')

    # 증폭 배수 계산 (Low 대비 High)
    sub_low = df[(df['Traffic_Density'] == 0) & (df['Weather'] == weather) & (df['Road_Condition'] == road)]
    sub_high = df[(df['Traffic_Density'] == 2) & (df['Weather'] == weather) & (df['Road_Condition'] == road)]
    if len(sub_low) >= 3 and len(sub_high) >= 3:
        r_low = sub_low['Accident_Severity'].mean()
        r_high = sub_high['Accident_Severity'].mean()
        if r_low > 0:
            amplify = f'{r_high / r_low:.1f}x'
        else:
            amplify = '-'
    else:
        amplify = 'N/A'

    print(f'{weather:<10} {road:<20} {rates[0]:<12} {rates[1]:<12} {rates[2]:<12} {amplify:<10}')

print("\n※ 증폭배수 = High(2) 심각비율 / Low(0) 심각비율")
print("※ N/A = 해당 조합의 데이터가 3건 미만으로 신뢰도 부족")
