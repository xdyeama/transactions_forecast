import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import timedelta

# Torch Dataset
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==== Model ====
class SalesRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3, model_type="LSTM"):
        super().__init__()
        if model_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Load & preprocess ====
df = pd.read_csv("./data_processed/data.csv", parse_dates=["date_time"])
df = df.sort_values("date_time")

# Aggregate daily sales
df["date"] = df["date_time"].dt.date
daily_sales = df.groupby("date").size().reset_index(name="sales")

# Create sequences of last N days
def create_daily_sequences(data, input_days=14):
    X, y = [], []
    for i in range(len(data) - input_days):
        X.append(data[i:i+input_days])
        y.append(data[i+input_days])
    return np.array(X), np.array(y)

sales_daily = daily_sales["sales"].values
X, y = create_daily_sequences(sales_daily, input_days=30)

# Torch Dataset
dataset = SalesDataset(X, y)  # Same dataset class, but y will be scalar
dataset.y = dataset.y.unsqueeze(-1)  # For regression

train_size = int(len(dataset) * 0.8)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model for 1-step daily prediction
model_strategic = SalesRNN(output_size=1, model_type="GRU").to(device)

optimizer = torch.optim.Adam(model_strategic.parameters(), lr=0.03)
criterion = nn.MSELoss()

for epoch in range(100):
    model_strategic.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model_strategic(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

    model_strategic.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model_strategic(X_batch)
            val_loss += criterion(preds, y_batch).item()
    print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model_strategic.state_dict(), "./models/model_versions/strategic_rnn.pth")
