import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

LAYER_SIZES = [784, 128, 10]       # Arkitektur: Input -> Skjult -> Output
ACTIVATION = nn.ReLU               # nn.ReLU, nn.Tanh, nn.LeakyReLU, etc.
EPOCHS = 3                         # Antal gennemløb af datasættet
BATCH_SIZE = 64                    # Batchstørrelse under træning
LEARNING_RATE = 0.001              # Hvor store skridt optimizer tager
DATA_DIR = "./data"                # Hvor MNIST downloades til
OUTPUT_WEIGHTS_FILE = "mlp_weights.pt"   # Gemt PyTorch model

def build_model(layer_sizes, activation_fn):
    """Bygger et dynamisk sekventielt netværk ud fra listen af lag."""
    layers = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fn())
    layers.pop()  # Fjern sidste aktivering -> vi vil have rå logits
    return nn.Sequential(*layers)


def main():
    # 1. Hent og forbered MNIST
    transform = transforms.ToTensor()  # Skalerer pixels til float [0.0, 1.0]
    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # 2. Initialiser model, tabsfunktion og optimizer
    model = build_model(LAYER_SIZES, ACTIVATION)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starter træning: {' -> '.join(map(str, LAYER_SIZES))}")

    # 3. Træningsloop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            # Fladgør billedet fra (B, 1, 28, 28) til (B, 784)
            x = images.view(images.size(0), -1)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} | Gennemsnitlig Loss: {avg_loss:.4f}")

    # 4. Evaluer nøjagtighed på test-sættet
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.view(images.size(0), -1)
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = (correct / len(test_data)) * 100
    print(f"\nTest Nøjagtighed: {acc:.2f}%")

    # 5. Gem modellen
    torch.save(model.state_dict(), OUTPUT_WEIGHTS_FILE)
    print(f"Model gemt til: {OUTPUT_WEIGHTS_FILE}")