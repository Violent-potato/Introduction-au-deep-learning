import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CardioDataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Sortie binaire [0, 1]
        )

    def forward(self, x):
        return self.net(x)


# Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=batch['features'].shape[1], hidden_size=128).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 1e-4
l2_lambda = 1e-3

for epoch in range(10):
    model.train()
    for batch in train_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

        # Calcul de la pénalité L1 (somme des valeurs absolues des poids)
        l1_penalty = sum(p.abs().sum() for p in model.parameters())

        # Calcul de la pénalité L2 (somme des carrés des poids)
        l2_penalty = sum(p.pow(2).sum() for p in model.parameters())

        # Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        loss.backward()
        optimizer.step()