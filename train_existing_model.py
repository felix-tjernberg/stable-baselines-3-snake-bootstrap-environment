print('Remember to start tensorboard with commands "pipenv shell" then "tensorboard --logdir=logs/train_sessions/"')
from snake_gym_environment import SnakeGymEnvironment, SnakeGymEnvironmentCallback
from stable_baselines3 import PPO
import time
import os

start_time = int(time.time())


# Change these values before running script
model_type = PPO
model_name_prefix = "ppo_snake_gym_environment_v1"
total_timesteps_per_episode = 100000
starting_episode = 1  # 1 * 100000 = 100000
existing_model_name = "ppo_snake_gym_environment_v1_1654791166_1_100000"
saved_model_path = f"models/saved_models/{existing_model_name}"
model_number = 1  # Number for the model to be trained (I recommend that you train at least 3 models in parallel, results vary a lot in reinforcement learning)


model_name = f"{model_name_prefix}_{start_time}_{model_number}"
models_directory = f"models/train_sessions/{model_name}/"
logs_directory = f"logs/train_sessions/"
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)


environment = SnakeGymEnvironment()
environment.reset()
loaded_model = model_type.load(saved_model_path, environment)


while True:
    starting_episode += 1
    loaded_model.learn(
        total_timesteps=total_timesteps_per_episode,
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}",
        callback=SnakeGymEnvironmentCallback(),
    )
    loaded_model.save(f"{models_directory}/{model_name}_{total_timesteps_per_episode*starting_episode}")
