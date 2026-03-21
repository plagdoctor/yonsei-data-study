# ============================================================
# [Step 4] 데이터셋 상세 검토
# ============================================================
# 각 변수의 고유값 분포, 결측치 현황, 이상치 등을
# 더 상세히 확인합니다.
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# 1) 결측치 현황 요약
# ----------------------------------------------------------
print("=" * 60)
print("[1] 결측치 현황")
print("=" * 60)
missing = df.isnull().sum()  # 각 컬럼별 결측치 수 계산
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)  # 결측치 비율(%) 계산
missing_df = pd.DataFrame({
    '결측치 수': missing,
    '결측치 비율(%)': missing_pct
})
# 결측치가 있는 변수만 필터링하여 출력
print(missing_df[missing_df['결측치 수'] > 0])
print(f"\n전체 변수 중 결측치 포함 변수: {(missing > 0).sum()}개")

# ----------------------------------------------------------
# 2) 범주형 변수 고유값 분포 확인
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("[2] 범주형 변수 고유값 분포")
print("=" * 60)

# object 타입인 컬럼만 선택
cat_cols = df.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    print(f"\n--- {col} ---")
    print(f"고유값 수: {df[col].nunique()}개")
    print(df[col].value_counts())  # 각 고유값의 빈도수 출력

# ----------------------------------------------------------
# 3) 수치형 변수 고유값 및 이상치 확인
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("[3] 수치형 변수 고유값 및 이상치 확인")
print("=" * 60)

# 수치형 컬럼 선택
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in num_cols:
    print(f"\n--- {col} ---")
    print(f"고유값 수: {df[col].nunique()}개")
    print(f"최솟값: {df[col].min()}, 최댓값: {df[col].max()}")
    print(f"평균: {df[col].mean():.2f}, 중앙값: {df[col].median():.2f}")

    # IQR 기반 이상치 탐지
    Q1 = df[col].quantile(0.25)  # 1사분위수
    Q3 = df[col].quantile(0.75)  # 3사분위수
    IQR = Q3 - Q1                # 사분위 범위
    lower = Q1 - 1.5 * IQR       # 이상치 하한
    upper = Q3 + 1.5 * IQR       # 이상치 상한
    outliers = df[(df[col] < lower) | (df[col] > upper)]  # 이상치 필터링
    print(f"IQR 기반 이상치 수: {len(outliers)}개 (하한: {lower}, 상한: {upper})")

# ----------------------------------------------------------
# 4) 타겟 변수(Accident_Severity) 분포 확인
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("[4] 타겟 변수(Accident_Severity) 분포")
print("=" * 60)
print(df['Accident_Severity'].value_counts())  # 각 클래스별 빈도수
print(f"\n사고 심각(1) 비율: {df['Accident_Severity'].mean() * 100:.2f}%")
