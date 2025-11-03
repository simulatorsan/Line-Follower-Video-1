import gymnasium as gym
import line_follower_v0
import pygame

ENV_NAME = "line_follower_v0"
enable_assist = True

env = gym.make(
    f'my_gym_envs/{ENV_NAME}', render_mode="human",
    # f'my_gym_envs/{ENV_NAME}', render_mode=None,
    sensor_grid = (4, 6),
    track="oval",
    max_steps=500,
    # hitbox=20,
)


# Initialize pygame
pygame.init()
window = pygame.display.set_mode((200, 200))  # dummy window for event handling
pygame.display.set_caption("MountainCar Controller")

clock = pygame.time.Clock()

while True:
    # state, _ = env.reset(seed=23)
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = 1  # default: center

        if enable_assist:
            vals = state[:4]
            if vals[0]:
                action = 0
            elif vals[3]:
                action = 2

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2


        # Step environment
        state, reward, terminated, truncated, _ = env.step(action)
        # print(state.astype(int), state.shape)
        total_reward += reward
        done = terminated or truncated

        # clock.tick(30)  # Limit to 30 FPS

    print(f"Episode finished! Total reward: {total_reward:.2f}")
