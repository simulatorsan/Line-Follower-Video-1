import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

model_names = [a[:-4] for a in os.listdir("for_video/saved_models/") if a.endswith(".pth")]

model_names.sort(key=int)

for it in tqdm(model_names):
    file_path = f"for_video/saved_models/{it}.pth"
    loaded = torch.load(file_path, weights_only=False)
    rewards_per_episode = loaded["rewards_per_episode"]
    test_episodes = loaded["test_episodes"]
    test_rewards = loaded["test_rewards"]
    episode = loaded["episode"]
    
    

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
    
    
    
    
    
    
    # Set the style to dark_background
    plt.style.use('dark_background')

    # Create the plot
    fig, ax = plt.subplots()

    x_range = range(len(rewards_per_episode))
    x_range_smoothed = range(len(smoothed))

    # Plot the data with dark-mode friendly colors
    ax.plot(x_range, rewards_per_episode, color='cyan', alpha=0.3)
    ax.plot(x_range_smoothed, smoothed, color='cyan', label="Train Reward")
    ax.plot(test_episodes, test_rewards, 'yellow', label="Test Reward")

    # Set labels with a larger font size
    ax.set_xlabel("Episode", color='white', fontsize=15)
    ax.set_ylabel("Reward", color='white', fontsize=15)

    # Set tick parameters with a larger font size
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    # Set spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    # Set legend with a larger font size
    ax.legend(loc='upper left', fontsize=15)

    # Set grid
    ax.grid(True, color='gray', linestyle='--')

    # Set background color
    fig.set_facecolor('black')
    ax.set_facecolor('black')
    
    
    
    
    
    
    
    
    
    plt.savefig(f"for_video/graphs/{it}.png", dpi=175, bbox_inches='tight')
    plt.close()
