import pandas as pd
import os.path as op


def read_data(clr='white'):
    file_name = 'winequality-' + clr + '.csv'
    file_dir = op.abspath(op.join(op.pardir, 'datasets', file_name))

    with open(file_dir, 'r') as dataset:
        df = pd.read_csv(dataset)

    return df


if __name__ == "__main__":
    df = read_data()
