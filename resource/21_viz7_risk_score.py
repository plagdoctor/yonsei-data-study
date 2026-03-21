# ============================================================
# [시각화 7] 복합 위험 점수 시각화 — 위험 요인 중첩 효과
# ============================================================
# 분석 질문: 위험 요인이 몇 개 겹칠 때 심각 사고 비율이
#            급등하는 임계점이 존재하는가?
# 관련 변수: Weather, Road_Condition, Traffic_Density, Speed_Limit,
#            Accident_Severity
# 시각화 유형: 계단형 바 차트 + 누적 구성 분석
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

# ----------------------------------------------------------
# 1) 위험 요인 플래그 생성 (각 요인별 0 또는 1)
# ----------------------------------------------------------
# 위험 요인 1: 나쁜 날씨 (Clear가 아닌 모든 날씨)
df['Risk_Weather'] = (df['Weather'] != 'Clear').astype(int)

# 위험 요인 2: 위험한 도로 상태 (Dry가 아닌 모든 상태)
df['Risk_Road'] = (df['Road_Condition'] != 'Dry').astype(int)

# 위험 요인 3: 높은 교통 혼잡도 (Traffic_Density == 2)
df['Risk_Density'] = (df['Traffic_Density'] == 2).astype(int)

# 위험 요인 4: 높은 제한 속도 (80 초과)
df['Risk_Speed'] = (df['Speed_Limit'] > 80).astype(int)

# 복합 위험 점수 (0~4): 위험 요인 개수의 합
df['Risk_Score'] = (df['Risk_Weather'] + df['Risk_Road'] +
                    df['Risk_Density'] + df['Risk_Speed'])

# ----------------------------------------------------------
# 2) 위험 점수별 통계 계산
# ----------------------------------------------------------
risk_stats = df.groupby('Risk_Score').agg(
    total=('Accident_Severity', 'count'),      # 전체 건수
    severe=('Accident_Severity', 'sum'),        # 심각 사고 건수
    rate=('Accident_Severity', 'mean')          # 심각 사고 비율
).reset_index()
risk_stats['rate_pct'] = (risk_stats['rate'] * 100).round(1)
risk_stats['pct_of_total'] = (risk_stats['total'] / len(df) * 100).round(1)

# ----------------------------------------------------------
# 3) 시각화: 2×2 대시보드
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

fig.suptitle('Compound Risk Score Analysis\nHow Many Risk Factors Overlap → Severe Accident Rate',
             fontsize=15, fontweight='bold', y=1.02)

# --- (3-1) 메인: 위험 점수별 심각 사고 비율 (계단형 바 차트) ---
colors_gradient = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

bars = axes[0, 0].bar(
    risk_stats['Risk_Score'], risk_stats['rate_pct'],
    color=[colors_gradient[i] for i in risk_stats['Risk_Score']],
    edgecolor='black', linewidth=0.8, width=0.6
)

# 각 바 위에 비율과 건수 표시
for bar, (_, row) in zip(bars, risk_stats.iterrows()):
    axes[0, 0].text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
        f'{row["rate_pct"]}%\n({int(row["severe"])}/{int(row["total"])}건)',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

# 전체 평균선
avg_rate = df['Accident_Severity'].mean() * 100
axes[0, 0].axhline(y=avg_rate, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0, 0].text(4.3, avg_rate + 1, f'Overall Avg: {avg_rate:.1f}%',
                 fontsize=9, color='gray', fontweight='bold')

# 임계점 강조 (50% 기준선)
axes[0, 0].axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 0].text(4.3, 51, '50% Threshold', fontsize=8, color='red', alpha=0.7)

axes[0, 0].set_title('Severe Rate by Risk Score', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Number of Risk Factors (0~4)', fontsize=10)
axes[0, 0].set_ylabel('Severe Accident Rate (%)', fontsize=10)
axes[0, 0].set_xticks(range(5))
axes[0, 0].set_xticklabels(['0\n(Safe)', '1', '2', '3', '4\n(Most\nDangerous)'])
axes[0, 0].set_ylim(0, max(risk_stats['rate_pct']) * 1.25)

# --- (3-2) 건수 분포: 위험 점수별 데이터 비중 ---
bars2 = axes[0, 1].bar(
    risk_stats['Risk_Score'], risk_stats['total'],
    color=[colors_gradient[i] for i in risk_stats['Risk_Score']],
    edgecolor='black', linewidth=0.8, width=0.6, alpha=0.8
)
for bar, (_, row) in zip(bars2, risk_stats.iterrows()):
    axes[0, 1].text(
        bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
        f'{int(row["total"])}건\n({row["pct_of_total"]}%)',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )
axes[0, 1].set_title('Data Distribution by Risk Score', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Risk Factors (0~4)', fontsize=10)
axes[0, 1].set_ylabel('Number of Accidents', fontsize=10)
axes[0, 1].set_xticks(range(5))
axes[0, 1].set_ylim(0, max(risk_stats['total']) * 1.25)

# --- (3-3) 개별 위험 요인의 기여도 ---
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
    factor_rates.append({
        'factor': label,
        'exposed': exposed,
        'not_exposed': not_exposed,
        'diff': exposed - not_exposed
    })

factor_df = pd.DataFrame(factor_rates).sort_values('diff', ascending=True)

# 수평 쌍봉 바 차트
y_pos = range(len(factor_df))
bars_safe = axes[1, 0].barh(
    [y - 0.2 for y in y_pos], factor_df['not_exposed'],
    height=0.35, label='Without Risk Factor',
    color='#3498db', edgecolor='black', linewidth=0.5, alpha=0.7
)
bars_risk = axes[1, 0].barh(
    [y + 0.2 for y in y_pos], factor_df['exposed'],
    height=0.35, label='With Risk Factor',
    color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.7
)

# 차이 표시
for y, (_, row) in zip(y_pos, factor_df.iterrows()):
    axes[1, 0].text(
        row['exposed'] + 1, y + 0.2,
        f'+{row["diff"]:.1f}%p',
        va='center', fontsize=9, fontweight='bold', color='#c0392b'
    )

axes[1, 0].set_yticks(list(y_pos))
axes[1, 0].set_yticklabels(factor_df['factor'])
axes[1, 0].set_title('Individual Risk Factor Impact', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Severe Accident Rate (%)', fontsize=10)
axes[1, 0].legend(loc='lower right', fontsize=9)

# --- (3-4) 위험 점수별 위험 요인 구성 비율 (스택 바) ---
# 각 Risk Score에서 어떤 위험 요인이 얼마나 활성화되었는지
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
    axes[1, 1].bar(
        risk_composition.index, vals, bottom=bottom,
        label=label, color=factor_colors[i],
        edgecolor='white', linewidth=0.5, width=0.6
    )
    # 각 세그먼트에 비율 표시 (충분히 큰 경우만)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 15:  # 15% 이상일 때만 표시
            axes[1, 1].text(
                risk_composition.index[j], b + v/2,
                f'{v:.0f}%', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white'
            )
    bottom += vals

axes[1, 1].set_title('Risk Factor Composition by Score', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Risk Score', fontsize=10)
axes[1, 1].set_ylabel('Factor Activation Rate (%)', fontsize=10)
axes[1, 1].set_xticks(range(5))
axes[1, 1].legend(loc='upper left', fontsize=8, ncol=2)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4) 결과 요약 출력
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("복합 위험 점수 분석 결과 요약")
print("=" * 70)
print("\n[위험 요인 정의]")
print("  1. Bad Weather: 맑음(Clear)이 아닌 날씨")
print("  2. Bad Road: 건조(Dry)가 아닌 도로 상태")
print("  3. High Density: 교통 혼잡도 High(2)")
print("  4. High Speed: 제한 속도 80km/h 초과")

print(f"\n{'점수':<8} {'건수':<10} {'비율':<10} {'심각 건수':<10} {'심각 비율':<12} {'위험 수준'}")
print("-" * 65)
for _, row in risk_stats.iterrows():
    score = int(row['Risk_Score'])
    level = ['안전', '주의', '경고', '위험', '극도위험'][score]
    marker = ['🟢', '🟡', '🟠', '🔴', '🚨'][score]
    print(f'  {score}     {int(row["total"]):<10} {row["pct_of_total"]}%     '
          f'{int(row["severe"]):<10} {row["rate_pct"]}%       {marker} {level}')

# 임계점 분석
score_2plus = df[df['Risk_Score'] >= 2]
score_3plus = df[df['Risk_Score'] >= 3]
print(f"\n[임계점 분석]")
print(f"  위험 요인 2개 이상: 심각 비율 {score_2plus['Accident_Severity'].mean()*100:.1f}% (n={len(score_2plus)})")
print(f"  위험 요인 3개 이상: 심각 비율 {score_3plus['Accident_Severity'].mean()*100:.1f}% (n={len(score_3plus)})")
