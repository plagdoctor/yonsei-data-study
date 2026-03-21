# ============================================================
# [Full EDA Report — Part 2/3] 상관관계 + 카이제곱 + 상호작용
# ============================================================
# Part 1에서 생성한 pdf 객체에 이어서 페이지를 추가합니다.
# ============================================================

# ============================================================
# PAGE 6: 피어슨 상관계수 히트맵 + 다중공선성
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
page_header(fig, 'Page 6. Correlation Analysis',
            'Pearson Correlation (left) & Multicollinearity Check (right)')

# (6-1) 히트맵
corr_m = df[num_cols].corr().round(4)
sns.heatmap(corr_m, annot=True, fmt='.4f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=axes[0],
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
axes[0].set_title('Pearson Correlation Heatmap', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# (6-2) Driver_Age vs Experience 산점도
for s in [0, 1]:
    sub = df[df['Accident_Severity']==s]
    axes[1].scatter(sub['Driver_Age'], sub['Driver_Experience'], c=SEV[s],
                     label=SEV_LBL[s], alpha=0.35, s=25, edgecolors='white', linewidth=0.3)
z = np.polyfit(df['Driver_Age'], df['Driver_Experience'], 1)
p = np.poly1d(z)
xr = np.linspace(df['Driver_Age'].min(), df['Driver_Age'].max(), 100)
rv = df['Driver_Age'].corr(df['Driver_Experience'])
axes[1].plot(xr, p(xr), '--', color='black', lw=2, alpha=0.7, label=f'r = {rv:.4f}')
axes[1].set_title('Driver Age vs Experience\n(Multicollinearity)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Driver Age'); axes[1].set_ylabel('Driver Experience (years)')
axes[1].legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 6 저장 완료")

# ============================================================
# PAGE 7: 타겟과의 연관 강도 종합 (Pearson r + Cramer's V)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
page_header(fig, 'Page 7. Variable Association Strength with Target',
            "|Pearson r| for numeric, Cramer's V for categorical")

# 수치형: |Pearson r|
num_assoc = {c: abs(df[c].corr(df['Accident_Severity'])) for c in num_cols if c != 'Accident_Severity'}
# 범주형: Cramer's V
cat_assoc = {}
for col in cat_cols:
    ct = pd.crosstab(df[col], df['Accident_Severity'])
    chi2, pv, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum(); md = min(ct.shape) - 1
    cat_assoc[col] = np.sqrt(chi2 / (n * md))

all_a = {**num_assoc, **cat_assoc}
adf = pd.DataFrame({'Var': list(all_a.keys()), 'Assoc': list(all_a.values()),
                     'Type': ['Numeric']*len(num_assoc) + ['Categorical']*len(cat_assoc)})
adf = adf.sort_values('Assoc', ascending=True)

# (7-1) 수평 바 차트
bc = ['#e74c3c' if t == 'Categorical' else '#3498db' for t in adf['Type']]
bars = axes[0].barh(adf['Var'], adf['Assoc'], color=bc, edgecolor='black', lw=0.5, height=0.6)
for bar, v in zip(bars, adf['Assoc']):
    axes[0].text(bar.get_width()+0.008, bar.get_y()+bar.get_height()/2,
                  f'{v:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
axes[0].axvline(0.1, color='gray', ls=':', lw=1, alpha=0.6)
axes[0].axvline(0.3, color='gray', ls=':', lw=1, alpha=0.6)
axes[0].legend(handles=[Patch(fc='#3498db', ec='black', label='Numeric (|r|)'),
                         Patch(fc='#e74c3c', ec='black', label="Categorical (Cramer's V)")],
               loc='lower right', fontsize=9)
axes[0].set_title('All Variables — Association Strength', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Association'); axes[0].set_xlim(0, 0.7)

# (7-2) 카이제곱 검정 결과 표
chi_data = []
for col in cat_cols:
    ct = pd.crosstab(df[col], df['Accident_Severity'])
    chi2, pv, dof, _ = stats.chi2_contingency(ct)
    n = ct.sum().sum(); md = min(ct.shape) - 1
    cv = np.sqrt(chi2 / (n * md))
    sig = '✔ Yes' if pv < 0.05 else '✘ No'
    pstr = f'{pv:.2e}' if pv < 0.001 else f'{pv:.4f}'
    chi_data.append([col, f'{chi2:.1f}', pstr, f'{cv:.4f}', sig])

axes[1].axis('off')
tbl = axes[1].table(cellText=chi_data,
                     colLabels=['Variable', 'Chi²', 'p-value', "Cramer's V", 'Significant?'],
                     loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#34495e'); cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f8f9fa')
    # 유의한 행 강조
    if r > 0 and c == 4:
        txt = cell.get_text().get_text()
        if '✔' in txt:
            cell.set_text_props(color='#c0392b', fontweight='bold')
axes[1].set_title('Chi-Square Test Results (α=0.05)', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 7 저장 완료")

# ============================================================
# PAGE 8: Weather × Road_Condition 상호작용
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
page_header(fig, 'Page 8. Interaction: Weather × Road Condition',
            'Frequency (left) & Severe Rate % (right)')

wo = ['Rainy','Stormy','Snowy','Foggy','Clear']
ro = ['Wet','Icy','Under Construction','Dry']

cc = pd.crosstab(df['Weather'], df['Road_Condition']).reindex(index=wo, columns=ro)
sns.heatmap(cc, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Cross-tabulation (Frequency)', fontsize=12, fontweight='bold')

cs = (df.groupby(['Weather','Road_Condition'])['Accident_Severity'].mean()*100).unstack()
cs = cs.reindex(index=wo, columns=ro)
sns.heatmap(cs, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.5, ax=axes[1],
            vmin=0, vmax=100, cbar_kws={'label': 'Severe Rate (%)'})
axes[1].set_title('Severe Accident Rate (%)', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 8 저장 완료")

# ============================================================
# PAGE 9: Traffic_Density × Speed_Limit 상호작용
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
page_header(fig, 'Page 9. Interaction: Traffic Density × Speed Limit',
            'Box Plot (left) & Severe Rate by Speed Group × Density (right)')

dp = df.copy()
dp['TD'] = dp['Traffic_Density'].map(TD_LBL)
dp['Sev'] = dp['Accident_Severity'].map(SEV_LBL)
sns.boxplot(data=dp, x='TD', y='Speed_Limit', hue='Sev',
            palette=[SEV[0], SEV[1]], ax=axes[0],
            order=['Low (0)','Medium (1)','High (2)'])
axes[0].set_title('Speed Limit by Density & Severity', fontsize=12, fontweight='bold')
axes[0].legend(title='Severity', fontsize=9)

dp['SG'] = pd.cut(dp['Speed_Limit'], bins=[0,50,80,110],
                    labels=['Low(<=50)','Mid(51-80)','High(81-110)'])
ir = dp.groupby(['SG','Traffic_Density'])['Accident_Severity'].mean()*100
ip = ir.unstack(); ip.columns = [TD_LBL[c] for c in ip.columns]
ip.plot(kind='bar', ax=axes[1], color=[TD_CLR[0],TD_CLR[1],TD_CLR[2]],
        edgecolor='black', lw=0.5)
for cont in axes[1].containers:
    axes[1].bar_label(cont, fmt='%.1f%%', fontsize=8, padding=2)
axes[1].set_title('Severe Rate: Speed × Density', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Speed Group'); axes[1].set_ylabel('Severe Rate (%)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='Traffic Density', fontsize=9); axes[1].set_ylim(0, 100)

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 9 저장 완료")

# ============================================================
# PAGE 10: Traffic_Density 증폭 효과 — 패싯 히트맵
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
page_header(fig, 'Page 10. Traffic Density Amplification Effect',
            'Severe Rate by Weather × Road Condition at Low / Medium / High Density')

for idx, td in enumerate([0, 1, 2]):
    ds = df[df['Traffic_Density']==td]
    cr = (ds.groupby(['Weather','Road_Condition'])['Accident_Severity'].mean()*100).unstack()
    cn = ds.groupby(['Weather','Road_Condition'])['Accident_Severity'].count().unstack()
    aw = [w for w in wo if w in cr.index]
    ar = [r for r in ro if r in cr.columns]
    cr = cr.reindex(index=aw, columns=ar)
    cn = cn.reindex(index=aw, columns=ar)
    ann = cr.copy().astype(str)
    for w in aw:
        for r in ar:
            rv2 = cr.loc[w,r] if not pd.isna(cr.loc[w,r]) else np.nan
            cv2 = cn.loc[w,r] if not pd.isna(cn.loc[w,r]) else 0
            ann.loc[w,r] = f'{rv2:.0f}%\n(n={int(cv2)})' if not pd.isna(rv2) and cv2>0 else '-'
    sns.heatmap(cr, annot=ann, fmt='', cmap='RdYlGn_r', vmin=0, vmax=100,
                linewidths=0.8, linecolor='white', ax=axes[idx],
                cbar=(idx==2), cbar_kws={'label':'Severe %','shrink':0.8} if idx==2 else {},
                annot_kws={'fontsize': 9})
    sr2 = ds['Accident_Severity'].mean()*100
    axes[idx].set_title(f'Density: {TD_LBL[td]}\n(n={len(ds)}, Avg:{sr2:.1f}%)',
                         fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Road Condition'); axes[idx].set_ylabel('Weather' if idx==0 else '')
    if idx > 0: axes[idx].set_yticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.92])
pdf.savefig(fig); plt.close()
print("✔ Page 10 저장 완료")
print("\n>>> Part 2 완료 — Part 3 셀을 실행하세요 >>>")
