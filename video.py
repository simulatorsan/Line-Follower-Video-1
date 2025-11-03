import gymnasium as gym
import torch
import torch.nn as nn
from models import DQN
import numpy as np
import os
import argparse

def evaluate_model(
    model, env_name,
    sensor_grid,
    track,
    max_steps,
    hitbox,
    x_spacing,
    y_spacing,
    episodes,
    verbose=False,
    output_path=None,
    save_threshold=None,
    invert_waypoints=None,
    invert_colours=None
):
    # If an output path is provided, we must use 'rgb_array' to collect frames
    render_mode = "rgb_array" if output_path else "human"
    
    env = gym.make(
        f'my_gym_envs/{env_name}', render_mode=render_mode,
        sensor_grid=sensor_grid,
        track=track,
        max_steps=max_steps,
        hitbox=hitbox,
        x_spacing=x_spacing,
        y_spacing=y_spacing,
        invert_waypoints=invert_waypoints,
        invert_colours=invert_colours
    )
    
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frames = []

        if output_path:
            frames.append(env.render())

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated

            if output_path:
                frames.append(env.render())

        total_rewards.append(total_reward)
        if verbose: print(f"Episode {ep+1}: Total Reward = {total_reward}")

        if output_path and save_threshold is not None:
            if total_reward >= save_threshold:
                run_data = np.array(frames)
                np.save(output_path, run_data)
                if verbose:
                    print(f"Score >= {save_threshold}. Saved run data to {output_path} with shape {run_data.shape}")
            else:
                if verbose:
                    print(f"Score < {save_threshold}. Discarding run data.")

    env.close()
    return np.mean(total_rewards)

if __name__ == "__main__":
    import line_follower_v0

    parser = argparse.ArgumentParser(description="Evaluate a DQN model for the line follower environment.")
    parser.add_argument("--model_path", type=str, default="dqn_linefollower.pth", help="Path to the trained model file.")
    parser.add_argument("--save_dir", type=str, default="run_data", help="Directory to save the run data.")
    parser.add_argument("--filename", type=str, default="run_1.npy", help="Filename for the saved run data.")
    parser.add_argument("--save_threshold", type=float, default=0.0, help="Minimum reward threshold to save the episode data.")
    parser.add_argument("--invert_waypoints", action="store_true", help="Invert waypoints during evaluation.")
    parser.add_argument("--invert_colours", action="store_true", help="Invert track colors during evaluation.")
    
    args = parser.parse_args()
    # print(f"{args.invert_waypoints=}, {args.invert_colours=}")

    os.makedirs(args.save_dir, exist_ok=True)
    assert os.path.exists(args.model_path), f"Model file not found: {args.model_path}"
    
    loaded_model = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=False)
    sensor_grid = loaded_model["sensor_grid"]
    hitbox = loaded_model["hitbox"]
    track = loaded_model["track"]

    policy_net = DQN(
        sensor_grid[0] * sensor_grid[1],
        loaded_model["action_dim"],
        loaded_model["hidden_dim"],
        loaded_model["hidden_layers"]
    )
    policy_net.load_state_dict(loaded_model["state_dict"])
    policy_net.eval()

    output_filename = os.path.join(args.save_dir, args.filename)
    avg_reward = evaluate_model(
        policy_net,
        "line_follower_v0",
        sensor_grid,
        track,
        max_steps=loaded_model["max_steps"],
        hitbox=hitbox,
        x_spacing=loaded_model["x_spacing"],
        y_spacing=loaded_model["y_spacing"],
        episodes=1,
        verbose=True,
        output_path=output_filename,
        save_threshold=args.save_threshold,
        invert_waypoints=args.invert_waypoints,
        invert_colours=args.invert_colours
    )
