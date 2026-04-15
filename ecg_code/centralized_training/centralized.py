import torch
from torch.utils.data import ConcatDataset, DataLoader
from ecg_code.federated_training.task import Net, load_data, train, test # Imports the model and the functions to load data and train/test the model


# This is the centralized training (with combined datasets)
# In the real world, this would be like a database with the datasets
# from both hospitals.
# Is expected to have the best results. And would break all multiple GDPR rules.

def main():
    print("Starting centralized learning (with combined datasets)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Loading data
    print("\nLoading PTB-XL (Client 0)...")
    trainloader_0, testloader_0, _ = load_data(0)
    
    print("\nLoading PTB-DB (Client 1)...")
    trainloader_1, testloader_1, _ = load_data(1)
    
    # Combining data to one centralized dataset
    print("\nCombining datasets...")
    combined_trainset = ConcatDataset([trainloader_0.dataset, trainloader_1.dataset])
    combined_testset = ConcatDataset([testloader_0.dataset, testloader_1.dataset])
    
    trainloader = DataLoader(combined_trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(combined_testset, batch_size=32)
    
    print(f"Combined number of patients in training_dataset: {len(combined_trainset)}")
    print(f"Combined number of patients in test_dataset: {len(combined_testset)}")
    
    # Readying InceptionTime-model
    net = Net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Train - 10 Rounds for proper evaluation turnaround
    rounds = 10 
    
    print("\nStarting training and evaluation...")
    for round in range(1, rounds + 1):
        print(f"\nRound: {round} ...")
        
        # Running training
        train(net, trainloader, optimizer, epochs=1, device=device)
        
        # Evaluate
        if round == rounds:
            # Apples-to-Apples Testing: Centralized MUST take the exact same 73-patient exam as the other models!
            test(net, testloader_0, device=device, save_path="cent_client0")
            loss, accuracy, f1, auc = test(net, testloader_1, device=device, save_path="cent_client1")
        else:
            loss, accuracy, f1, auc = test(net, testloader, device=device)
        
        print(f"Average Loss: {loss:.4f}")
        print(f"Accuracy:     {accuracy:.4f} ({(accuracy*100):.1f}%)")
        print(f"F1-Score:     {f1:.4f}")
        print(f"ROC AUC:      {auc:.4f}")
        
    print("\nCOMPLETED!")

if __name__ == "__main__":
    main()
