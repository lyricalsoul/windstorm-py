import json
import traceback
from websockets.legacy.client import connect
from prediction.model import MCPredictor, get_criterion
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from prediction.utils import load_data
import asyncio


should_train = False
should_test = False

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

early_stopping = EarlyStopper(patience=10, min_delta=0)

# load the data from the csv file7
data = load_data()

# split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# create the model
model = MCPredictor()
optimizer = model.get_optimizer()
criterion = get_criterion()

def calculate_validation_loss():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for index, row in test_data.iterrows():
            # get the input and output
            input_data = model.return_inputs(row)
            output_data = model.return_outputs(row)

            outputs = model(input_data)
            loss = criterion(outputs, output_data)
            total_loss += loss.item()

    return total_loss / len(test_data)

if should_train:
    # train the model: we want to predict the temperature and humidity (TEMPERATURA DO AR - BULBO SECO, HORARIA (C) and UMIDADE RELATIVA DO AR, HORARIA (%)
    epochs = 100
    for epoch in range(epochs):
        running_loss = 0.0
        for index, row in train_data.iterrows():
            # get the input and output
            input_data = model.return_inputs(row)
            output_data = model.return_outputs(row)

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
else:
    # load the model
    model.load_state_dict(torch.load('../models/mc_predictor.pth'))

# run model test
if should_test:
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
            print(f'Input: {input_data}')
            print(f'Output: {outputs}, expected: {output_data}')
            if torch.all(torch.eq(outputs, output_data)):
                correct += 1

        print(f'Accuracy: {correct / total}')

    print('Finished training. Final accuracy: ', correct / total)

# asyncio because i am dumb and tried to use threading but it barely worked
async def handle(websocket):
    msg = await websocket.recv()

    decoded = json.loads(msg)
    if decoded['op'] == 'ai-request':
        print(f'Got request: {decoded}')
        pred = model.inaccurate_predict_weather(
            day_of_the_year=decoded['timeOfDay'],
            time=decoded['hour'],
            temp_max_hour=decoded['maxTemp'],
            temp_min_hour=decoded['minTemp'],
            humidity_max_hour=decoded['maxHumidity'],
            humidity_min_hour=decoded['minHumidity']
        )

        temp = pred[0]
        humidity = pred[1]

        # convert to 1 decicaml float
        temp = round(temp.item(), 1)
        humidity = round(humidity.item(), 1)

        print(f'Sending prediction: {pred} - temp: {temp}, humidity: {humidity}')
        await websocket.send(json.dumps({'op': 'ai-response', 'temp': temp, 'humidity': humidity}))
    elif decoded['op'] == 'inmet-alert':
        print(f'got inmet alert: {decoded['message']}')
    else:
        print(f'got unknown op {decoded["op"]}')

    await handle(websocket)

uri = 'ws://localhost:8765'

async def main():
    async for websocket in connect(uri):
        try:
            await handle(websocket)
        except Exception as ex:
            print(f'Error: {ex}')
            traceback.print_exc()

asyncio.run(main())