import torch
from ecg_code.federated_training.task_equal import Net, load_data, train, test # Imports the equal model

# This is the centralized training (with individual datasets).
# This is how most hospitals do it today.
# One dataset per hospital. No sharing of data or models.
# Is expected to have the worst results.

def train_individual_model(client_id, rounds=20):
    # Client 0: PTB-XL
    # Client 1: PTB-DB
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting learning for client {client_id}")
    
    # Last inn kun data for denne spesifikke klienten
    print(f"Loading data for client {client_id}...")
    trainloader, testloader, num_examples = load_data(client_id)
    print(f"Training on {num_examples['trainset']} patients, testing on {num_examples['testset']} patients.")
    
    # Readying InceptionTime-model
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Train and evaluate (only on own data)
    for round in range(1, rounds + 1):
        print(f"\nRound {round}")
        train(net, trainloader, optimizer, epochs=20, device=device)
        
        if round == rounds:
            loss, accuracy, f1, auc = test(net, testloader, device=device, save_path=f"equal_iso_client{client_id}")
        else:
            loss, accuracy, f1, auc = test(net, testloader, device=device)
        
        print(f"Loss:      {loss:.4f}")
        print(f"Accuracy:  {accuracy:.4f} ({(accuracy*100):.1f}%)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        
    print(f"\nCompleted training for client {client_id}!\n")

def main():
    print("Starting centralized learning (with individual datasets)\n")
    
    # Running independent training for Client 0 (PTB-XL)
    print("Loading PTB-XL (Client 0)")
    train_individual_model(client_id=0, rounds=20)
    
    # Running independent training for Client 1 (PTBDB)
    print("Loading PTB-DB (Client 1)")
    train_individual_model(client_id=1, rounds=20)
    
    print("\nAll isolated training is complete!")

if __name__ == "__main__":
    main()
