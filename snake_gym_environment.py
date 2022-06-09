from gym import spaces, Env
from snake_game.snake_game_agent_version import SnekGame
from stable_baselines3.common.callbacks import BaseCallback
from numpy import array, float32, inf


class SnakeGymEnvironment(Env):
    def __init__(self):
        super(SnakeGymEnvironment, self).__init__()
        self.action_space = spaces.Discrete(4)  # Action number required for predictions of left, right, up, down
        self.observation_space = spaces.Box(
            low=-inf,
            high=inf,
            shape=(5,),  # The observation shape is a in this case a one dimensional vector with 5 observations
            dtype=float32,
        )
        self.snake_game = SnekGame()

    def step(self, action):
        self.snake_game.agent_action = action
        self.snake_game.game_tick()

        # Rewards the agent for eating food and punish for game over
        if self.snake_game.game_over:
            self.reward = self.snake_game.score - 10
        else:
            self.reward = self.snake_game.score

        return (
            self.observe_game_state(),  # Observation
            self.reward,
            self.snake_game.game_over,  # If the episode is done
            {"game_score": self.snake_game.score},  # Info object
        )

    def reset(self):
        self.reward = 0
        self.snake_game.reset_game()
        return self.observe_game_state()  # In reset we only return the observation

    def observe_game_state(self):
        snake_head = self.snake_game.head
        food = self.snake_game.food
        food_delta_x = snake_head.x - food.x
        food_delta_y = snake_head.y - food.y

        return array(
            [
                snake_head.x,
                snake_head.y,
                food_delta_x,
                food_delta_y,
                len(self.snake_game.body),
            ],
            dtype=float32,
        )  # The one dimensional vector with 5 observations

    def render(self, mode="human"):
        self.snake_game.display_update = True  # This will start to render the game graphically for when we are running or visually inspecting the game

    def close(self):
        self.snake_game.quit_game()


# Callback that records game score for each step to the logger object so we can see how the agent is doing in tensorbard
class SnakeGymEnvironmentCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SnakeGymEnvironmentCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("game_score", self.locals["infos"][0]["game_score"])
        return True
