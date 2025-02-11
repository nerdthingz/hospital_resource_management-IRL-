import gymnasium as gym
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import minimize
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

data=pd.read_csv("/content/first_8_rows.csv")
data = data.drop('allocate_to_another_floor', axis=1)
features=data.drop(["priority_floor","medicine_reallocation"],axis=1).to_numpy()
floor=data["priority_floor"].to_numpy()
medicine=data["medicine_reallocation"].to_numpy()
expert_actions = []
for f, m in zip(floor, medicine):
    expert_actions.append((f, m))

expert_states=[]
for i in range(features.shape[0]):
    expert_states.append(features[i])


#Combine state and action 
expert_trajectories = [[(state, action) for state,action in zip(expert_states,expert_actions)]]

class MedicalDistributionEnv(gym.Env):
    def __init__(self, feature_size, num_floors, max_medicine):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(feature_size,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([num_floors, max_medicine + 1])

        self.state = None
        self.num_floors = num_floors
        self.max_medicine = max_medicine
        self.expert_states = expert_states

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random_index = self.np_random.choice(len(self.expert_states))
        self.state = self.expert_states[random_index]
        return self.state, {}
      
# --- MaxEntIRL implementation(finding the reward function) ---
class MaxEntIRL:
    def __init__(self, env, expert_trajectories):
        self.env = env
        self.expert_trajectories = expert_trajectories
        self.feature_size = env.observation_space.shape[0] + np.sum(env.action_space.nvec)

    def _state_action_features(self, state, action):
        action_features = []
        for i, a in enumerate(action):
            one_hot = np.zeros(self.env.action_space.nvec[i])
            one_hot[a] = 1
            action_features.extend(one_hot)
        return np.concatenate([state, action_features])

    def _reward_function(self, weights, state, action):
        return np.dot(weights, self._state_action_features(state, action))

    def _policy(self, weights, state):
        action_space_size = np.prod(self.env.action_space.nvec)
        probabilities = np.zeros(action_space_size)
        z = 0
        for i in range(action_space_size):
            action = []
            temp_idx = i
            for n in reversed(self.env.action_space.nvec):
                action.insert(0, temp_idx % n)
                temp_idx //= n

            reward = self._reward_function(weights, state, action)
            probabilities[i] = np.exp(reward)
            z += probabilities[i]

        return probabilities / z if z > 0 else np.ones(action_space_size) / action_space_size

    def _negative_log_likelihood(self, weights):
        nll = 0
        for trajectory in self.expert_trajectories:
            for state, action in trajectory:
                probabilities = self._policy(weights, state)
                action_index = 0
                temp = 1
                for i, a in enumerate(reversed(action)):
                    action_index += a * temp
                    temp *= self.env.action_space.nvec[len(action) - 1 - i]

                if probabilities[action_index] > 0:
                    nll -= np.log(probabilities[action_index])
                else:
                    return np.inf
        return nll

    def learn_reward_function(self):
        initial_weights = np.random.rand(self.feature_size)
        with tqdm(total=1, desc="Optimization Progress") as pbar:
            result = minimize(self._negative_log_likelihood, initial_weights, method='L-BFGS-B', callback=lambda xk: pbar.update(1))
        return result.x

# Instantiate environment and train MaxEntIRL
env = MedicalDistributionEnv(feature_size=features.shape[1], num_floors=np.max(floor) + 1, max_medicine=np.max(medicine))
irl_model = MaxEntIRL(env, expert_trajectories)
learned_reward_weights = irl_model.learn_reward_function()

# --- Learned reward function ---
def learned_reward(state, action, weights):
    action_features = []
    for i, a in enumerate(action):
        one_hot = np.zeros(env.action_space.nvec[i])
        one_hot[a] = 1
        action_features.extend(one_hot)
    state_action_features = np.concatenate([state, action_features])
    return np.dot(weights, state_action_features)

# --- Use the learned reward in an RL algorithm (PPO) ---
class LearnedRewardEnv(MedicalDistributionEnv):
    def step(self, action):
        reward = learned_reward(self.state, action, learned_reward_weights)
        done = True
        return self.state, reward, done, False, {}

learned_reward_env = LearnedRewardEnv(feature_size=features.shape[1], num_floors=np.max(floor) + 1, max_medicine=np.max(medicine))
learned_reward_env = DummyVecEnv([lambda: learned_reward_env])

model = PPO("MlpPolicy", learned_reward_env, verbose=1)
model.learn(total_timesteps=10000)

# Save the learned reward function weights
np.save("learned_reward_weights.npy", learned_reward_weights)
print("Learned reward function weights saved to learned_reward_weights.npy")

# Save the trained policy
model.save("trained_policy")
print("Trained policy saved to trained_policy.zip")
