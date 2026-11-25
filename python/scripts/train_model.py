# in: --dataset (mnist, fashion or kmnist) images and labels, LAYER_SIZES: list[int], training settings
# out: trained checkpoint (.pt), float weight/bias CSVs, printed test accuracy, in python/models/ (MNIST) or python/models/<dataset>/
# Trains a float MLP and exports its parameters for the Scala loader.

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

LAYER_SIZES = [784, 128, 10]       # arch: Input -> hidden -> Output
ACTIVATION = nn.ReLU               # nn.ReLU, nn.Tanh, nn.LeakyReLU, etc.
EPOCHS = 3                         # n runthroughs of dataset
BATCH_SIZE = 64                    # Batchssize during training
LEARNING_RATE = 0.001              # how big steps a optimizer takes
# MNIST paths (python/tests and benchmark_cpu.py import them); main() picks the folders for --dataset
PYTHON_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PYTHON_DIR / "data"
OUTPUT_WEIGHTS_FILE = PYTHON_DIR / "models/mlp_weights.pt"
CSV_DIR = PYTHON_DIR / "models/csv"
RESULTS_DIR = PYTHON_DIR / "results"
QUANTIZED_DIR = PYTHON_DIR.parent / "generator/generated/quantized"
# --dataset -> torchvision class; each one downloads to DATA_DIR/<class name>/raw
DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST, "kmnist": datasets.KMNIST}


def model_dir(dataset):
    """MNIST keeps the committed python/models/; another dataset gets python/models/<dataset>/."""
    return PYTHON_DIR / "models" if dataset == "mnist" else PYTHON_DIR / "models" / dataset


def build_model(layer_sizes, activation_fn):
    """Bygger et dynamisk sekventielt netværk ud fra listen af lag."""
    layers = []
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation_fn())
    layers.pop()  # del last activation -> we want raw logits
    return nn.Sequential(*layers)


def main():
    # 0. Which dataset: python -m scripts.train_model --dataset fashion (or make train DATASET=fashion)
    parser = argparse.ArgumentParser(description="Train the float MLP and export its weights as CSVs")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    dataset = parser.parse_args().dataset
    weights_file = model_dir(dataset) / "mlp_weights.pt"
    csv_dir = model_dir(dataset) / "csv"

    # 1. download and prepare the dataset
    transform = transforms.ToTensor()  # scales pixel to float [0.0, 1.0]
    train_data = DATASETS[dataset](DATA_DIR, train=True, download=True, transform=transform)
    test_data = DATASETS[dataset](DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # 2. Initialize model, loss function and optimizer
    model = build_model(LAYER_SIZES, ACTIVATION)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starter træning: {' -> '.join(map(str, LAYER_SIZES))}")

    # 3. trainingloop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            # flatten pic from (B, 1, 28, 28) to (B, 784)
            x = images.view(images.size(0), -1)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS} | Gennemsnitlig Loss: {avg_loss:.4f}")

    # 4. Evaluate 
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.view(images.size(0), -1)
            preds = model(x).argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = (correct / len(test_data)) * 100
    print(f"\nTest Nøjagtighed: {acc:.2f}%")

    # 5. save model
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_file)
    print(f"Model gemt til: {weights_file}")

    # 6. Save param + bias as csv
    # The CSVs feed the Scala export and the hardware, which only implement ReLU hidden layers
    if ACTIVATION is not nn.ReLU:
        raise ValueError(f"CSV export needs ACTIVATION = nn.ReLU, got {ACTIVATION.__name__}; the checkpoint is saved")
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, param in model.named_parameters():
        filename = csv_dir / f"{name.replace('.', '_')}.csv"
        np.savetxt(filename, param.detach().numpy().T, delimiter=",") #[in_features, out_features]
        print(f"Gemte {filename}")

if __name__ == "__main__":
    main()
