from flask import Flask, jsonify, request
import numpy as np
import csv


import json
from datetime import date
import joblib
import torch
from torch import nn

import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import string
import random
import threading
import time
import datetime


# HOST = config.settings['host']
# MASTER_KEY = config.settings['master_key']
# DATABASE_ID = config.settings['database_id']
# EVENTCONTAINER_ID = config.settings['eventcontainer_id']
# COWCONTAINER_ID = config.settings['cowcontainer_id']
# CONTAINER_ID = config.settings['container_id']


app = Flask(__name__)
# client = cosmos_client.CosmosClient(HOST, {'masterKey': MASTER_KEY}, user_agent="CosmosDBDotnetQuickstart",
#                                     user_agent_overwrite=True)

# try:
#     db = client.get_database_client(DATABASE_ID)

#     try:
#         cowcontainer = database.get_container_client(COWCONTAINER_ID)
#         eventcontainer = database.get_container_client(EVENTCONTAINER_ID)

#     except exceptions.CosmosHttpResponseError as e:
#         print('\nError caught: {0}'.format(e.message))

# except exceptions.CosmosHttpResponseError as e:
#     print('\nError caught: {0}'.format(e.message))

# finally:
#     print("\nConnected")

filename = 'model1'
model = NN()
model.load_state_dict(torch.load('rua'), strict=False)
model.eval()


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # Number of input features is 6.
        self.layer_1 = nn.Linear(6, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


@app.route('/')
def index():
    return "Cow Mastitis Predictor!"


@app.route('/api/cow/predict/', methods=['GET'])
def predict():
    x = request.json
    y = model(torch.tensor([x]))
    pred = torch.round(torch.sigmoid(y.float()))
    result = "mastitis" if pred == 1 else 'healthy'
    pred = jsonify({'condition': result})
    return pred
    # except:
    #     return "input error"


@app.route('/api/c/', methods=['GET'])
def maps():
    return 2*2


@app.route('/api/cow/<id>', methods=['POST'])
def addCowDay(cid):
    row = request.json.get("id")
    response = updateRow(cowcontainer, cid, row)
    return response.status


def query(container, key):
    item = list(container.query_items(
        query="SELECT * FROM c WHERE c.key=@key",
        parameters=[
            {"name": "@key", "value": key}
        ],
        enable_cross_partition_query=True
    ))
    return item


def addNewCow(container, body):
    response = container.create_item(body=sales_order)
    return response


def updateRow(container, key, row):
    item = list(container.query_items(
        query="SELECT * FROM c WHERE c.key=@key",
        parameters=[
            {"name": "@key", "value": key}
        ],
        enable_cross_partition_query=True
    ))
    obj = json.loads(item)
    newItem = json.dumps(obj.append(row))
    response = container.replace_item(
        item=item, body=newItem)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
