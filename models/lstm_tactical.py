import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import timedelta

# ==== Load & preprocess ====
df = pd.read_csv("./data_processed/data.csv", parse_dates=["date_time"])
df = df.sort_values("date_time")

# Aggregate to 10-minute bins
df["time_bin"] = df["date_time"].dt.floor("10T")
sales_agg = df.groupby("time_bin").size().reset_index(name="sales")

# Create lag features (sequence)
def create_sequences(data, input_steps=12, output_steps=3):  # 12*10min=2hrs, output=3 steps = 30min
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

sales_series = sales_agg["sales"].values
X, y = create_sequences(sales_series, input_steps=12, output_steps=3)

# Torch Dataset
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SalesDataset(X, y)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

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
model = SalesRNN(model_type="GRU").to(device)

# ==== Training ====
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            val_loss += criterion(preds, y_batch).item()
    print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "./models/model_versions/tactical_rnn.pth")
