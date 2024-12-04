from prediction.model import MCPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from prediction.tester import run_model_test
from prediction.utils import load_data


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


early_stopping = EarlyStopper(patience=10, min_delta=1)

# load the data from the csv file7
data = load_data()

# we must drop the index column
# before, print what column we are dropping
# print the column length
print(len(data.columns))

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

input_size = 19 - 2
output_size = 2

# create the model
model = MCPredictor(input_size, 16, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = torch.nn.MSELoss()


def calculate_validation_loss():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for index, row in test_data.iterrows():
            # get the input and output
            input_data = torch.tensor(row.drop(
                ['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']).values.astype(
                np.float32), dtype=torch.float32)
            output_data = torch.tensor(
                row[['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)',
                     'UMIDADE RELATIVA DO AR, HORARIA (%)']].values.astype(
                    np.float32), dtype=torch.float32)

            outputs = model(input_data)
            loss = criterion(outputs, output_data)
            total_loss += loss.item()

    return total_loss / len(test_data)


# train the model: we want to predict the temperature and humidity (TEMPERATURA DO AR - BULBO SECO, HORARIA (C) and UMIDADE RELATIVA DO AR, HORARIA (%)
epochs = 100
for epoch in range(epochs):
    running_loss = 0.0
    for index, row in train_data.iterrows():
        # get the input and output
        input_data = torch.tensor(row.drop(
            ['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']).values.astype(
            np.float32), dtype=torch.float32)
        output_data = torch.tensor(
            row[['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)']].values.astype(
                np.float32), dtype=torch.float32)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_data)
        loss = criterion(outputs, output_data)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(train_data)}')

    validation_loss = calculate_validation_loss()
    print(f'Validation loss: {validation_loss}')
    if early_stopping.early_stop(validation_loss):
        print("Early stopping NOW!")
        break

# save the model
torch.save(model.state_dict(), '../models/mc_predictor.pth')

run_model_test()

# now, infer temperature and humidity for an arbitrary input
