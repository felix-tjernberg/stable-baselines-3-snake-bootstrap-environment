print('Remember to start tensorboard with commands "pipenv shell" then "tensorboard --logdir=logs/train_sessions/"')
from snake_gym_environment import SnakeGymEnvironment, SnakeGymEnvironmentCallback
from stable_baselines3 import PPO
import time
import os


# Change these values before running script
model_type = PPO
model_name_prefix = "ppo_snake_gym_environment_v1"
total_timesteps_per_episode = 100000
model_number = 1  # Number of the model to be trained (I recommend that you train at least 3 models in parallel, results vary a lot in reinforcement learning)


start_time = int(time.time())
model_name = f"{model_name_prefix}_{start_time}_{model_number}"
models_directory = f"models/train_sessions/{model_name}/"
logs_directory = f"logs/train_sessions/"
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)


environment = SnakeGymEnvironment()
environment.reset()
model = PPO("MlpPolicy", environment, verbose=1, tensorboard_log=logs_directory)


episode = 0
while True:
    episode += 1
    model.learn(
        total_timesteps=total_timesteps_per_episode,
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}",
        callback=SnakeGymEnvironmentCallback(),
    )
    model.save(f"{models_directory}/{model_name}_{total_timesteps_per_episode*episode}")
