from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context, Metrics

# Running the server for federated learning
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply metrics of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    auc_scores = [num_examples * m["roc_auc"] for num_examples, m in metrics if "roc_auc" in m]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metrics (weighted average)
    result = {
        "accuracy": sum(accuracies) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples)
    }
    if auc_scores:
        result["roc_auc"] = sum(auc_scores) / sum(examples)
        
    return result

def server_fn(context: Context):
    # Setup for central server
    num_rounds = 3
    
    # Configure to demand exactly 2 clients for fit and evaluate
    strategy = FedAvg(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)