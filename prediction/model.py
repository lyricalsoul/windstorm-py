import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim, tensor

from prediction.utils import load_data

learning_rate = 1e-3
weight_decay = 1e-4

data = load_data()

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

def get_criterion():
    # use mean squared error so we can penalize bigger deviations on tensor values
    return nn.MSELoss()

class MCPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = 2
        self.input_size = 19 - self.output_size

        self.hidden_size = 32

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_optimizer(self):
       return optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # drop temp and humidity params since we want to predict them
    def return_inputs(self, row):
        return torch.tensor(row.drop(
            ['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']).values.astype(
            np.float32), dtype=torch.float32)

    # only return what we dropped since it's what we are predicting
    def return_outputs(self, row):
        return torch.tensor(
            row[['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']].values.astype(
                np.float32), dtype=torch.float32)

    def predict_weather (self, time, precipitation, atmospheric_pressure,
                         at_press_max, at_press_min, global_radiation, orvalho_temp,
                        temp_max_hour, temp_min_hour, orvalho_temp_max_hour, orvalho_temp_min_hour,
                        humidity_max_hour, humidity_min_hour, wind_direction, wind_max_gust, wind_speed, day_of_the_year):
        # order is DATA (YYYY-MM-DD);
        # HORA (UTC);
        # PRECIPITAÇÃO TOTAL, HORÁRIO (mm);
        # PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB);]
        # RESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB);
        # PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB);
        # RADIAÇÃO GLOBAL (kJ/m2);
        # TEMPERATURA DO AR - BULBO SECO, HORARIA (C);
        # TEMPERATURA DO PONTO DE ORVALHO (C);
        # TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (C);
        # TEMPERATURA MINIMA NA HORA ANT. (AUT) (C);
        # TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (C);
        # TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (C);
        # UMIDADE REL. MAX. NA HORA ANT. (AUT) (%);
        # UMIDADE REL. MIN. NA HORA ANT. (AUT) (%);
        # UMIDADE RELATIVA DO AR, HORARIA (%);
        # VENTO, DIREÇÃO HORARIA (graus);
        # VENTO, RAJADA MAXIMA (m/s);
        # VENTO, VELOCIDADE HORARIA (m/s);
        # DIA DO ANO
        input_tensor = tensor(
            [time, precipitation, atmospheric_pressure, at_press_max, at_press_min, global_radiation, orvalho_temp,
             temp_max_hour, temp_min_hour, orvalho_temp_max_hour, orvalho_temp_min_hour, humidity_max_hour,
             humidity_min_hour, wind_direction, wind_max_gust, wind_speed, day_of_the_year]).float()

        return self(input_tensor)

    def inaccurate_predict_weather (self, time, precipitation = 0, atmospheric_pressure = 997, at_press_max = 997.5, at_press_min = 996.6,
                                    global_radiation = 1554, orvalho_temp = 30, temp_max_hour = 31, temp_min_hour = 30, orvalho_temp_max_hour = 31,
                                    orvalho_temp_min_hour = 30, humidity_max_hour = 70, humidity_min_hour = 60, wind_direction = 90, wind_max_gust = 5, wind_speed = 3,
                                    day_of_the_year = 1):
        return self.predict_weather(time, precipitation, atmospheric_pressure, at_press_max, at_press_min, global_radiation, orvalho_temp,
                                    temp_max_hour, temp_min_hour, orvalho_temp_max_hour, orvalho_temp_min_hour, humidity_max_hour,
                                    humidity_min_hour, wind_direction, wind_max_gust, wind_speed, day_of_the_year)