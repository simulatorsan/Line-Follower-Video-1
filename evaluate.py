import gymnasium as gym
import torch
import torch.nn as nn
from models import DQN
import numpy as np

def evaluate_model(
    model, env_name, render_mode,
    sensor_grid,
    track,
    max_steps,
    hitbox,
    x_spacing,
    y_spacing,
    episodes,
    verbose=False
):
    env = gym.make(
        f'my_gym_envs/{env_name}', render_mode=render_mode,
        sensor_grid=sensor_grid,
        track=track,
        max_steps=max_steps,
        hitbox=hitbox,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        verbose=verbose
    )
    # env.metadata["render_fps"] = 5
    
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated

        total_rewards.append(total_reward)
        if verbose: print(f"Episode {ep+1}: Total Reward = {total_reward}")

    env.close()
    return np.mean(total_rewards)

if __name__ == "__main__":
    import line_follower_v0
    ENV_NAME = "line_follower_v0"
    MODEL_PATH = "dqn_linefollower.pth"

    import os
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"

    loaded_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    sensor_grid = loaded_model["sensor_grid"]
    hitbox = loaded_model["hitbox"]

    max_steps = loaded_model["max_steps"]
    # max_steps = 500
    track = loaded_model["track"]
    # track = "rounded_square"

    policy_net = DQN(
        sensor_grid[0]*sensor_grid[1],
        loaded_model["action_dim"],
        loaded_model["hidden_dim"],
        loaded_model["hidden_layers"]
    )
    policy_net.load_state_dict(loaded_model["state_dict"])
    policy_net.eval()

    for run in range(10):
        avg_reward = evaluate_model(
            policy_net,
            ENV_NAME,
            "human",
            sensor_grid,
            track,
            max_steps,
            hitbox,
            x_spacing=loaded_model["x_spacing"],
            y_spacing=loaded_model["y_spacing"],
            episodes=1,
            verbose=True
        )
