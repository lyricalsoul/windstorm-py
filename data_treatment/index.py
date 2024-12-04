import numpy as np
import pandas as pd

# although the readme is in portuguese, we speak english when coding here. get with the program!
# we load the data from the csv file, show basics and then define functions to treat the data
raw_data = pd.read_csv('../raw_data/semitreated/MERGED_INMET_SERIES_2013-2014_SEMITREATED-2.csv',
                     sep=';', decimal=',', delimiter=None, encoding='utf-8')

raw_data['DATA (YYYY-MM-DD)'] = pd.to_datetime(raw_data['DATA (YYYY-MM-DD)'], format='%d/%m/%Y')

raw_data = raw_data.set_index('DATA (YYYY-MM-DD)')

print(raw_data.head())

# on column HORA (UTC), INMET used to report the tour in HH:MM format, but they've now changed it to "HHMM UTC".
# this steps replaces all the new format to the old one.
def fix_hour(data):
    data['HORA (UTC)'] = (data['HORA (UTC)']
                          .apply(lambda x: x.replace(' UTC', ''))
                          .apply(lambda x: x[:2] + ':' + x[2:] if len(x) == 4 else x))
    return

# this function reads 10k random values from the specified column, removes any -9999 or blank values and then generates the mean
def get_mean(data, column, sample_size = 10_000):
    random_values = data[column].sample(sample_size)
    random_values = random_values[random_values != -9999 & random_values.notnull()]
    return random_values.mean()

# this function replaces all -9999 and null values on a column with the specified value
def replace_null(data, column, value):
    data[column] = data[column].replace(-9999, value)
    data[column] = data[column].fillna(value)
    return

# smart fill uses the mean of all values of the day in a 6h period to fill the null values
def smart_fill(data, column):
    # first, we must filter everything on a day so we can get the mean from there. using the mean of all values in the column will result in wonky data
    grouped = data.groupby('DATA (YYYY-MM-DD)')[column]
    # now we iterate over each group
    for name, group in grouped:
        # we remove all -9999 and null values to get the mean
        mean = group[group != -9999 & group.notnull()].mean()
        # now we fill the null values with the mean
        group = group.fillna(mean, inplace=True)
        # also replace -9999 with the mean
        group = group.replace(-9999, mean)
        # finally, we replace the column with the new values
        data.loc[data['DATA (YYYY-MM-DD)'] == name, column] = group

# saves the data to a new csv file
def save_data(data):
    data.to_csv('../raw_data/treated/MERGED_INMET_SERIES_2013-2014_TREATED_2.csv', sep=';', decimal=',', encoding='utf-8')
    return

# the following columns can be safely filled with 0 (instead of -9999 and empty values)
simple_fill_required = ['RADIAÇÃO GLOBAL (kJ/m2)', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'VENTO, DIREÇÃO HORARIA (graus)', 'VENTO, RAJADA MAXIMA (m/s)', 'VENTO, VELOCIDADE HORARIA (m/s)']
smart_fill_required = [
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
    'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
    'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
    'UMIDADE RELATIVA DO AR, HORARIA (%)',
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (C)',
    'TEMPERATURA DO PONTO DE ORVALHO (C)',
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (C)',
    'TEMPERATURA MINIMA NA HORA ANT. (AUT) (°C)',
    'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (C)',
    'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (C)',
    'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
    'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)'
]

fix_hour(raw_data)
raw_data.replace(-9999, np.nan, inplace=True)
raw_data.replace(-9999.0, np.nan, inplace=True)
for column in simple_fill_required:
    replace_null(raw_data, column, 0)

for column in smart_fill_required:
    print(f'Filling column {column}...')
    # check if column exists
    if column in raw_data.columns:
        pass
    # smart_fill(raw_data, column)
# replace the date, in the dd/mm/yyyy format, with the day of the year. data is the index, so pull from there
raw_data['DIA DO ANO'] = raw_data.index.dayofyear
# replace the hour, in the hh:mm format, with the hour
raw_data['HORA (UTC)'] = raw_data['HORA (UTC)'].apply(lambda x: int(x.split(':')[0]))

# export the data to a new csv file
raw_data.interpolate(method='time', inplace=True)
save_data(raw_data)