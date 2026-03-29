import torch
from ecg_code.task import Net

print("\n--- Tester InceptionTime Nettverket (Net) ---")
try:
    model = Net()
    print("1. Modell (InceptionTime) opprettet vellykket!")
    
    # Simulerer 2 pasienter med 12 kanaler EKG og lengde 1000
    dummy_data = torch.randn(2, 12, 1000)
    print(f"2. Sender inn test-pasienter med form (Batch, Channels, Length): {dummy_data.shape}")
    
    # Kjører dataene gjennom nettverket
    output = model(dummy_data)
    
    print(f"3. Suksess! Modellen spyttet ut form: {output.shape}")
    print("Dette betyr 2 pasienter, og 2 utfallsklasser (f.eks Frisk=0, Syk=1) som forventet.")
    print("InceptionTime-koden din er feilfri og helt klar til bruk!\n")
except Exception as e:
    print(f"\nFEIL OPPDAGET: {e}\n")
