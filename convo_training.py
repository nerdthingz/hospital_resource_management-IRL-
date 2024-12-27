import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

data = pd.read_csv("/kaggle/input/hospital-prject-data/time_series_data_final.csv")
X = data[['date', 'time', 'num_patients', 'news_score']].values
y_binary = data['more_patients'].values  # Binary classification target
y_regression = data['num_patients_admitted'].values  # Regression target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1) 
y_binary_tensor = torch.tensor(y_binary, dtype=torch.float32).unsqueeze(1)
y_reg_tensor = torch.tensor(y_regression, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_binary_tensor, y_reg_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc_shared = nn.Linear(16 * 4, 32)  
        self.fc_binary = nn.Linear(32, 1)
        self.fc_regression = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) 
        shared = self.fc_shared(x)
        binary_out = torch.sigmoid(self.fc_binary(shared))
        reg_out = self.fc_regression(shared)
        return binary_out, reg_out

model = MultiTaskModel()
criterion_binary = nn.BCELoss()
criterion_regression = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_binary_batch, y_reg_batch in data_loader:
        optimizer.zero_grad()
        binary_out, reg_out = model(X_batch)
        loss_binary = criterion_binary(binary_out, y_binary_batch)
        loss_regression = criterion_regression(reg_out, y_reg_batch)
        loss = loss_binary + loss_regression
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "conv1d_multi_task_model.pth")
joblib.dump(scaler, "scaler.pkl")
