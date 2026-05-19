from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedProx
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
    # How many communication rounds between clients and global server
    num_rounds = 10
    
    def evaluate_config(server_round: int):
        return {"is_final_round": server_round == num_rounds}
    
    # Configure to demand exactly 2 clients for fit and evaluate
    strategy = FedProx(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=evaluate_config,
        proximal_mu=0.1,    # Added FedProx proximal term
                            # proximal_mu says how much we trust the global model
                            # if proximal_mu is 0 we dont trust the global model
                            # if proximal_mu is 1 we trust the global model completely
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)