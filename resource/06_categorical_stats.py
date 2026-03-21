# ============================================================
# [Step 6] 범주형 변수 기초통계 분석 및 표 정리
# ============================================================
# 범주형 변수에 대해 각 고유값의 빈도, 비율(%), 결측치 수를
# 변수별로 표 형태로 정리합니다.
# ============================================================

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# ----------------------------------------------------------
# 1) 범주형 변수 선택
# ----------------------------------------------------------
cat_cols = df.select_dtypes(include='object').columns.tolist()

# ----------------------------------------------------------
# 2) 변수별 빈도/비율/결측치 표 출력
# ----------------------------------------------------------
for col in cat_cols:
    print("=" * 60)
    print(f"▶ {col}")
    print("=" * 60)

    # 각 고유값의 빈도수 계산 (결측치 제외)
    freq = df[col].value_counts()

    # 각 고유값의 비율(%) 계산 (전체 840건 기준)
    ratio = (df[col].value_counts() / len(df) * 100).round(2)

    # 결측치 수 계산
    missing_count = df[col].isnull().sum()

    # 표(DataFrame) 형태로 결합
    cat_table = pd.DataFrame({
        '빈도': freq,
        '비율(%)': ratio
    })

    # 합계 행 추가
    cat_table.loc['합계(결측 제외)'] = [
        cat_table['빈도'].sum(),       # 빈도 합계
        cat_table['비율(%)'].sum()     # 비율 합계
    ]

    print(cat_table.to_string())
    print(f"\n결측치: {missing_count}건 ({(missing_count / len(df) * 100):.2f}%)")
    print(f"고유값 수: {df[col].nunique()}개")
    print()

# ----------------------------------------------------------
# 3) 범주형 변수 요약 총괄표
# ----------------------------------------------------------
print("=" * 60)
print("범주형 변수 요약 총괄표")
print("=" * 60)

# 총괄 요약 정보를 리스트로 수집
summary_list = []
for col in cat_cols:
    missing = df[col].isnull().sum()                    # 결측치 수
    n_unique = df[col].nunique()                        # 고유값 수
    top_value = df[col].mode()[0]                       # 최빈값
    top_freq = df[col].value_counts().iloc[0]           # 최빈값 빈도
    top_ratio = round(top_freq / len(df) * 100, 2)     # 최빈값 비율(%)

    summary_list.append({
        '변수명': col,
        '고유값 수': n_unique,
        '최빈값': top_value,
        '최빈값 빈도': top_freq,
        '최빈값 비율(%)': top_ratio,
        '결측치 수': missing,
        '결측치 비율(%)': round(missing / len(df) * 100, 2)
    })

# DataFrame으로 변환 후 출력
summary_df = pd.DataFrame(summary_list).set_index('변수명')
print(summary_df.to_string())
