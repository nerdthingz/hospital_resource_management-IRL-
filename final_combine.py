import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from fastapi import FastAPI
app = FastAPI()

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

model = PPO.load("/content/trained_policy.zip") 

def time_series_pred(input_array,ppo_array, model_weights="/content/conv1d_multi_task_model.pth", scaler="/content/scaler.pkl"):
    loaded_model = MultiTaskModel()
    loaded_model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu'))) 
    loaded_model.eval()
    loaded_scaler = joblib.load(scaler)
    input_array = np.array(input_array).reshape(1, -1)  
    input_scaled = loaded_scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(1)  
    with torch.no_grad():
        binary_prediction, regression_prediction = loaded_model(input_tensor)
    
    if binary_prediction>=0.70:
      action, _states = model.predict(ppo_array, deterministic=True)
      return action
    else:
      return None

@app.post("/predict/")
def predict(first_array,agent_array):
  sol=time_series_pred(first_array,agent_array)
  if sol:
    print("priority_floor :",sol[0],"medicine_amount:",sol[1])
