import torch
from flwr.app
from flwr.clientapp import ClientApp

# TODO: Importer din modell og dine load-funksjoner fra task.py
# from code.task import Net, load_data, train, test

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data"""