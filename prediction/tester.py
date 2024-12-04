from prediction.model import MCPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from prediction.utils import load_data

# load the model at ../models/mc_predictor.pth
input_size = 19 - 2
output_size = 2

# create the model
model = MCPredictor(input_size, 64, output_size)
model.load_state_dict(torch.load('../models/mc_predictor.pth'))

# load the data from the csv file
data = load_data()

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

def run_model_test():
    # test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for index, row in test_data.iterrows():
            # get the input and output
            input_data = torch.tensor(row.drop(
                ['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']).values.astype(
                np.float32), dtype=torch.float32)
            output_data = torch.tensor(
                row[['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']].values.astype(
                    np.float32), dtype=torch.float32)

            outputs = model(input_data)
            total += 1
            if torch.all(torch.eq(outputs, output_data)):
                correct += 1

    print(f'Accuracy: {correct / total}')