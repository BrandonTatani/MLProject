import pandas as pd
from settings import DATA_DIR

def concat_dataset():
    """Dataset was already split and kaggle load was not working"""

    df_1 = pd.read_csv(DATA_DIR / 'train.csv')
    df_2 = pd.read_csv(DATA_DIR / 'validation.csv')
    df_3 = pd.read_csv(DATA_DIR / 'test.csv')

    _df = pd.concat([df_1, df_2, df_3])
    _df.to_csv(DATA_DIR / 'dataset.csv', index=False)

if __name__ == '__main__':

    df = pd.read_csv(DATA_DIR / 'dataset.csv')
    print(df.shape)