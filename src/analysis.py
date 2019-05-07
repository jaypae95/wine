def data_analysis(df):
    high_quality = df[df['quality'] == 'high']

    # 1. 만족도가 높은 데이터에서 나타나는 특징 분석
    # 만족도가 high인 것만 뽑아서 ( df[df['quality']=='high']
    # 각 column들 하나 하나 분석
    # low, mid data에서도 비슷한 결과가 나오는 지 파악해야 하기 때문에 low, mid데이터와도 비교
    analysis_each_columns(df)

    # 2. 만족도가 높은 데이터에서 column들 간의 관계 분석
    analysis_high_quality(df)

    # 3. 전체 만족도와 column들 분석

    # 4. PCA


def analysis_each_columns(df):
    high_quality = df[df['quality'] == 'high']
    mid_quality = df[df['quality'] == 'mid']
    low_quality = df[df['quality'] == 'low']

    # your code


def analysis_high_quality(df):
    print(1)

