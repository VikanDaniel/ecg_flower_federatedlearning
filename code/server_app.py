from flwr.serverapp import ServerApp, Context
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server import ServerAppComponents

def server_fn(context: Context):
    """
    Oppsett av den sentrale serveren som samler sammen vektene fra alle klientene (aggregering).
    Denne funksjonen settes opp idet serveren starter.
    """
    
    # TODO: Les inn eventuelle parametere fra pyproject.toml via context 
    # (f.eks num_rounds = context.run_config.get("num-server-rounds", 3))
    num_rounds = 3
    
    # TODO: Velg konfigurasjonen din for runde-antall
    config = ServerConfig(num_rounds=num_rounds)
    
    # TODO: Opprett en aggregeringsstrategi, standardvalget er Federated Averaging (FedAvg). 
    # Denne håndterer korleis vektene gjennomsnittes og minimumantall klienter som kreves for trening.
    strategy = FedAvg(
        # TODO: Tilpass strategien din her, foreksempel:
        # fraction_fit=1.0,  Sier at 100% av klientene skal brukes for fit
        # min_available_clients=2  Starter ikke før 2 ekte klienter kobler til
    )
    
    # Returnerer oppsettet tilbake til rammeverket
    return ServerAppComponents(strategy=strategy, config=config)

# Oppretter Flower ServerApp basert på funksjonen over
app = ServerApp(server_fn=server_fn)