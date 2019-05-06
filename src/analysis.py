def data_analysis(df):
    test()
    high_quality = df[df['quality'] == 'high']
    print(high_quality)

    # 1. 만족도가 높은 데이터에서 나타나는 특징 분석
    # 만족도가 high인 것만 뽑아서 ( df[df['quality']=='high']
    # 각 column들 하나 하나 분석
    # low data에서도 비슷한 결과가 나오는 지 파악해야 하기 때문에 low데이터와도 비교
    # 그럴거면 그냥 high mid low를 0, 1, 2로 해서 숫자가 높아질수록 각 column들 비교하는 것이 나을지도 모르겠네요.

    # 2. 만족도가 높은 데이터에서 column들 간의 관계 분석

    # 3. 전체 만족도와 column들 분석

    # 4. PCA


def test():
    print('an')
