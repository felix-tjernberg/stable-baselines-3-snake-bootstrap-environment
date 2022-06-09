from snake_gym_environment import SnakeGymEnvironment
from stable_baselines3 import PPO


# Change these values before running script
model_path = f"models/saved_models/ppo_snake_gym_environment_v1_1654791166_1_100000"
game_speed = 30  # How fast we want the agent to appear when playing the game


environment = SnakeGymEnvironment()
environment.snake_game.GAME_SPEED = game_speed
environment.render()
loaded_model = PPO.load(model_path, environment)


# Runs 10 game sessions and print the score when session is done
for game_session in range(1, 11):
    observation = environment.reset()
    done = False
    while not done:
        action, _ = loaded_model.predict(observation)
        observation, reward, done, info = environment.step(action)
    print(f"Game {game_session} finished with a score of {environment.snake_game.score}")
environment.close()
