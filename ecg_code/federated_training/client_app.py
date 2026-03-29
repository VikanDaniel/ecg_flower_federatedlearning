import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# Use correct import from task
from ecg_code.federated_training.task import Net, load_data, train, test

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, testloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = dict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] Training...")
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        # Train for 1 epoch
        train(self.net, self.trainloader, optimizer, epochs=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] Evaluating...")
        self.set_parameters(parameters)
        loss, accuracy, f1, auc = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy), "f1_score": float(f1), "roc_auc": float(auc)}


def client_fn(context: Context):
    # Retrieve partition_id from the Context's NodeConfig
    partition_id = context.node_config["partition-id"]
    
    # Load model and data based on partition
    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    trainloader, testloader, _ = load_data(partition_id=partition_id)
    
    # Return FlowerClient
    return FlowerClient(partition_id, net, trainloader, testloader).to_client()

app = ClientApp(client_fn=client_fn)