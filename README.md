# hospital_resource_management-IRL

This project implements a comprehensive hospital resource management system. The system predicts future patient influx and optimizes resource allocation using 
inverse reinforcement learning (IRL) to find the reward function and the policy(apprenticeship learning). Below is an overview of the functionalities and methods used:

## Key Features

1. **Patient Influx Prediction**:
   - Predicts whether there will be an increase in the number of patients and estimates the magnitude of this increase.
   - Trained on past time series data .
   - Time series predictions are achieved using a 1D convolutional neural network (Conv1D).

2. **Resource Allocation Optimization**:
   - Implements an IRL algorithm to allocate resources such as room priority and medicine distribution efficientl.
   - The RL agent is trained on a small dataset provided as an example.

3. **Reinforcement Learning Training Process**:
   - Uses imitation learning (apprenticeship learning) to train the RL agent.
   - Derives the reward function from the data and subsequently learns the optimal policy.

## Files and Code Structure

- **Neural Network Training**:
  - The code for training the Conv1D neural network is located in the `convo_training file`.

- **Reinforcement Learning Training**:
  - The RL training process, including imitation learning and policy optimization, is implemented in `apprenticeship learning file`.

- **Final System Integration**:
  - The final combined system integrating prediction and resource allocation is found in `final_combine file` in which I have used fastapi to make prediction endpoints .

## Example Data

- Example data used for training the RL agent and prediction models is included in the repository.

This project serves as a robust solution for managing hospital resources efficiently while adapting to varying patient loads.
Feel free to further expand upon it


