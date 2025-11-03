import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, os
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

from evaluate import evaluate_model
from models import DQN
import line_follower_v0

ENV_NAME = "line_follower_v0"
EPISODES = 1000
GAMMA = 0.9
LR = 2e-5
BATCH_SIZE = 256
MEMORY_SIZE = 20_000
EPS_START = 1.0
EPS_END = 0.05
TARGET_UPDATE = 10   # update target network every N episodes
MODEL_PATH = "dqn_linefollower.pth"
PLOT_PATH = "linefollower_rewards.png"
EPS_DECAY = (EPS_END / EPS_START) ** (1/(0.2*EPISODES))
SEED = 23

# set the seed for both torch and numpy and everything
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

continue_training = True
sensor_grid = (4, 3)
# track = "oval"
# track = "hexagon"
track = "rounded_square_orig"
# track = "square_orig"
# track = "rounded_square"
# track = "square"
max_steps = 200
hitbox = 40
x_spacing = 40
y_spacing = 20

# hidden_dim=256
# hidden_layers=3

hidden_dim=32
hidden_layers=1

# Use GPU
device = torch.device("cpu")

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)

# Environment setup
env = gym.make(
    # f'gymnasium_env/{ENV_NAME}', render_mode="human",
    f'my_gym_envs/{ENV_NAME}', render_mode=None,
    sensor_grid = sensor_grid,
    track = track,
    max_steps = max_steps,
    hitbox = hitbox,
    x_spacing=x_spacing,
    y_spacing=y_spacing,
)

state_dim = env.observation_space.shape[0]   # (position, velocity)
action_dim = env.action_space.n              # 3 actions

policy_net = DQN(state_dim, action_dim, hidden_dim, hidden_layers).to(device)
target_net = DQN(state_dim, action_dim, hidden_dim, hidden_layers).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

start_episode = 0
stale_reward = None
rewards_per_episode = []
test_rewards = []
test_episodes = []

if continue_training and os.path.exists(MODEL_PATH):
    print("Continuing training from saved model.")
    loaded_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    policy_net.load_state_dict(loaded_model["state_dict"])
    target_net.load_state_dict(loaded_model["state_dict"])
    optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
    start_episode = loaded_model["episode"] + 1
    stale_reward = loaded_model["reward"]
    rewards_per_episode = loaded_model["rewards_per_episode"]
    test_rewards = loaded_model["test_rewards"]
    test_episodes = loaded_model["test_episodes"]

epsilon = max(EPS_START * EPS_DECAY ** start_episode, EPS_END)
target_net.eval()
# print(f"Epsilon decay rate: {EPS_DECAY:.4f}")

memory = ReplayBuffer(MEMORY_SIZE)

# Training Loop
progress_bar = tqdm(
    total=EPISODES,
    initial=start_episode,
    desc=f"Avg Reward (last 10): {stale_reward}, Epsilon: {epsilon:.2f}",
    dynamic_ncols=True
)
for episode in range(start_episode, EPISODES):
    state, _ = env.reset()
    total_reward = 0

    done = False
    while not done:
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train step if enough samples
        # if len(memory) >= BATCH_SIZE:
        if len(memory) >= (MEMORY_SIZE//2):
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rewards_per_episode.append(total_reward)
    
    if (episode+1) % TARGET_UPDATE == 0:
        # update target_network
        target_net.load_state_dict(policy_net.state_dict())

        # evaluate current model
        test_reward_mean = evaluate_model(
            policy_net,
            ENV_NAME,
            render_mode=None,
            sensor_grid=sensor_grid,
            track=track,
            max_steps=max_steps,
            hitbox=hitbox,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            episodes=10,
            verbose=False
        )
        test_rewards.append(test_reward_mean)
        test_episodes.append(episode)

        # make a plot
        window = 100
        if len(rewards_per_episode) >= window:
            # Pad the rewards array on both sides with terminal elements
            left_pad = window // 2
            right_pad = window - 1 - left_pad
            padded_rewards = np.concatenate([
                np.full(left_pad, rewards_per_episode[0]),
                rewards_per_episode,
                np.full(right_pad, rewards_per_episode[-1])
            ])
            smoothed = np.convolve(padded_rewards, np.ones(window)/window, mode='valid')
        else:
            smoothed = rewards_per_episode  # Not enough data for smoothing
        
        x_range = range(len(rewards_per_episode))
        plt.plot(x_range, rewards_per_episode, color='tab:blue', alpha=0.3)
        plt.plot(x_range, smoothed, color='tab:blue', label=f"Train Reward (smoothed over {window})")
        plt.plot(test_episodes, test_rewards, 'tab:orange', label="Test Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.title(f"({ENV_NAME}) Episode {episode+1}")
        plt.grid()
        # plt.ylim(0, 800)
        plt.savefig(f"rewards_plot.png")
        plt.close()
        
        
        
        # save checkpoint
        torch.save(
            {
                "state_dict": policy_net.state_dict(),
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": episode,
                "epsilon": epsilon,
                "rewards_per_episode": rewards_per_episode,
                "reward": np.mean(rewards_per_episode[-10:]) if len(rewards_per_episode) >= 10 else None,
                "test_rewards": test_rewards,
                "test_episodes": test_episodes,
                "sensor_grid": sensor_grid,
                "action_dim": action_dim,
                "track": track,
                "max_steps": max_steps,
                "hitbox": hitbox,
                "x_spacing": x_spacing,
                "y_spacing": y_spacing,
            },
            MODEL_PATH
        )

    # Update tqdm description every 100 episodes
    if (episode) % 10 == 0:
        avg_reward = np.mean(rewards_per_episode[-10:])
        progress_bar.set_description(f"Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")


    # Decay epsilon
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    progress_bar.update(1)

env.close()
progress_bar.close()
