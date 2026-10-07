import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Hent data (MNIST billeder normaliseret til [0.0, 1.0])
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=1000)

# 2. Definer modellen
def build_mlp(layer_sizes=[784, 128, 10]):
    layers = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    layers.pop()  # Fjern sidste ReLU så vi har rå logits til tabsfunktionen
    return nn.Sequential(*layers)

model = build_mlp([784, 128, 10])

# 3. Trænings-setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Træn 3 epoker
for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        # Fladgør billedet fra (64, 1, 28, 28) til (64, 784)
        x = images.view(images.size(0), -1)
        
        optimizer.zero_grad()       # Nulstil gamle hældninger
        output = model(x)           # Forward pass (matrix-gang)
        loss = criterion(output, labels)
        loss.backward()             # Beregn gradienter
        optimizer.step()            # Opdater vægte

    print(f"Epoch {epoch+1} færdig")

# 5. Evaluer på test-sættet
model.eval()
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        x = images.view(images.size(0), -1)
        preds = model(x).argmax(dim=1)
        correct += (preds == labels).sum().item()

print(f"Test Nøjagtighed: {correct / len(test_data) * 100:.2f}%")

# 6. Gem modellen
torch.save(model.state_dict(), "mlp_weights.pt")
print("Model gemt til mlp_weights.pt")