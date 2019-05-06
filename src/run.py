import pandas as pd


def read_data(clr='white'):
    file_name = 'winequality-' + clr + '.csv'
    with open(file_name, 'r') as dataset:
        df = pd.read_csv(dataset, names=[x for x in range(0, 10)])


if __name__ == "__main__":
    read_data()