import gymnasium as gym
import torch
import torch.nn as nn
from models import DQN
import numpy as np
import line_follower_v0, os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

def evaluate_model(
    model, env_name, render_mode,
    sensor_grid,
    track,
    max_steps,
    hitbox,
    x_spacing,
    y_spacing,
    episodes,
    verbose=False,
    invert_waypoints=None,
    invert_colours=None
):
    total_rewards = []
    for ep in range(episodes):
        env = gym.make(
            f'my_gym_envs/{env_name}', render_mode=render_mode,
            sensor_grid=sensor_grid,
            track=track,
            max_steps=max_steps,
            hitbox=hitbox,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            verbose=verbose,
            invert_waypoints=invert_waypoints,
            invert_colours=invert_colours
        )
        # env.metadata["render_fps"] = 5
    
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



models = []
episodes = []
for a in os.listdir("for_video/saved_models"):
    if a.endswith(".pth"):
        models.append(a)
        episodes.append(int(a[:-4]))
models.sort(key=lambda x: int(x[:-4]))
episodes.sort()

# invert_waypoints, invert_colours
perf_00 = []
perf_01 = []
perf_10 = []
perf_11 = []

for i, model_path in enumerate(tqdm(models)):
    loaded_model = torch.load(os.path.join("for_video/saved_models", model_path), map_location=torch.device('cpu'), weights_only=False)
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


    # 00
    perf_00.append(evaluate_model(
        policy_net,
        "line_follower_v0",
        None,
        sensor_grid,
        track,
        max_steps,
        hitbox,
        x_spacing=loaded_model["x_spacing"],
        y_spacing=loaded_model["y_spacing"],
        episodes=100,
        verbose=False,
        invert_waypoints=False,
        invert_colours=False
    ))

    # 01
    perf_01.append(evaluate_model(
        policy_net,
        "line_follower_v0",
        None,
        sensor_grid,
        track,
        max_steps,
        hitbox,
        x_spacing=loaded_model["x_spacing"],
        y_spacing=loaded_model["y_spacing"],
        episodes=100,
        verbose=False,
        invert_waypoints=False,
        invert_colours=True
    ))

    # 10
    perf_10.append(evaluate_model(
        policy_net,
        "line_follower_v0",
        None,
        sensor_grid,
        track,
        max_steps,
        hitbox,
        x_spacing=loaded_model["x_spacing"],
        y_spacing=loaded_model["y_spacing"],
        episodes=100,
        verbose=False,
        invert_waypoints=True,
        invert_colours=False
    ))

    # 11
    perf_11.append(evaluate_model(
        policy_net,
        "line_follower_v0",
        None,
        sensor_grid,
        track,
        max_steps,
        hitbox,
        x_spacing=loaded_model["x_spacing"],
        y_spacing=loaded_model["y_spacing"],
        episodes=100,
        verbose=False,
        invert_waypoints=True,
        invert_colours=True
    ))
    
    # print(episodes[:i+1], perf_00)


    # plt.figure(figsize=(10,6))
    # plt.plot(episodes[:i+1], perf_00, label="white  cw")
    # plt.plot(episodes[:i+1], perf_01, label="black  cw")
    # plt.plot(episodes[:i+1], perf_10, label="white ccw")
    # plt.plot(episodes[:i+1], perf_11, label="black ccw")
    # plt.plot(episodes[:i+1], np.mean([perf_00, perf_01, perf_10, perf_11], axis=0), label="Average", linestyle='--', color='black')
    # # plt.xticks(np.arange(50, episodes[i+1]+1, 50))
    # plt.xticks(np.arange(50, 800, 20))
    # # rotate the x labels
    # plt.xticks(rotation=90)
    # plt.xlabel("Training Episodes")
    # plt.ylabel("Average Reward")
    # plt.title("Model Performance under Different Inversion Settings")
    # plt.legend()
    # plt.grid()
    # plt.savefig("for_video/analyse.png")
    # plt.close()
    
    
    
    
    
    
    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axs[0].plot(episodes[:i+1], perf_01, label="black cw", color="tab:blue")
    axs[1].plot(episodes[:i+1], perf_11, label="black ccw", color="tab:orange")
    axs[2].plot(episodes[:i+1], perf_00, label="white cw", color="tab:green")
    axs[3].plot(episodes[:i+1], perf_10, label="white ccw", color="tab:red")

    for ax in axs:
        ax.set_ylabel("Avg Reward")
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Training Episodes")
    plt.xticks(np.arange(40, 801, 20), rotation=45)

    fig.suptitle("Model Performance under Different Inversion Settings")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("for_video/analyse100.png")
    plt.close()




