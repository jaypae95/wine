import pandas as pd
import os.path as op
from src import analysis as an
from src import prediction as pr

def read_data(clr='white'):
    file_name = 'winequality-' + clr + '.csv'
    file_dir = op.abspath(op.join(op.pardir, 'datasets', file_name))

    with open(file_dir, 'r') as dataset:
        df = pd.read_csv(dataset)

    return df


def quality_processing(df):
    high = [7, 8, 9, 10]
    mid = 6
    low = [0, 1, 2, 3, 4, 5]

    quality = df['quality']

    quality = quality.replace(high, 'high')
    quality = quality.replace(mid, 'mid')
    quality = quality.replace(low, 'low')

    df['quality'] = quality

    return df


if __name__ == "__main__":
    df = read_data()
    df = quality_processing(df)

    # data analysis
    an.test()

    # data prediction
    pr.test()
