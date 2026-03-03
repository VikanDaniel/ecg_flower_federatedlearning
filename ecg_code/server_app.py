from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context

def server_fn(context: Context):
    """
    Setup the central server configuring the Federated Averaging Strategy
    """
    num_rounds = 3
    
    # Federated Averaging strategy
    # Configure to demand exactly 2 clients for fit and evaluate
    strategy = FedAvg(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)