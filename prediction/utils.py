import pandas as pd

def load_data():
    data = pd.read_csv('../raw_data/treated/MERGED_INMET_SERIES_2013-2014_TREATED_2.csv',
                       index_col='DATA (YYYY-MM-DD)',
                       parse_dates=True,
                       sep=';', decimal=',', delimiter=None, encoding='utf-8')
    return data