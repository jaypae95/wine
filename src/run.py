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
    df = df[df['quality'] != 6]

    high = [7, 8, 9, 10]
    low = [0, 1, 2, 3, 4, 5]

    quality = df['quality']

    quality = quality.replace(high, 'high')
    quality = quality.replace(low, 'low')

    df['quality'] = quality
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = read_data()
    df = quality_processing(df)

    # data analysis
    an.data_analysis(df)

    # data prediction
    # pr.prediction(df)
