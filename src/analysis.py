import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_analysis(df):
    high_quality = df[df['quality'] == 'high']

    # 1. 만족도가 높은 데이터에서 나타나는 특징 분석
    # 만족도가 high인 것만 뽑아서 각 column들 하나 하나 분석
    # low data에서도 비슷한 결과가 나오는 지 파악해야 하기 때문에 low데이터와도 비교
    analysis_each_column(df)

    # 2. 만족도가 높은 데이터에서 column들 간의 관계 분석
    high_quality = df[df['quality'] == 'high']
    analysis_columns_by_correlation(high_quality)

    # 3. 전체 만족도와 column들 분석
    analysis_columns_by_correlation(df)

    # 4. PCA


def analysis_each_column(df):
    high_quality = df[df['quality'] == 'high']
    low_quality = df[df['quality'] == 'low']

    # your code


def analysis_columns_by_correlation(df):
    df = df.drop('quality', axis=1)  # quality 열 제거
    # plt.title로 제목 설정 가능(재활용성을 고려하여 설정안함)
    plt.figure(figsize=(12, 12))
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # sns.heatmap(데이터, 각 셀의 값 표시여부, 숫자표시, 색상, 반만 표시하기 위한 마스킹)
    sns.heatmap(data=corr, annot=True, fmt='.2f', linewidths=.5, cmap='plasma', mask=mask)
    plt.show()


def seperate_features_and_label(df):
    x = df.drop('quality', axis=1).values
    y = df['quality']

    return x, y