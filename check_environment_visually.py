from snake_gym_environment import SnakeGymEnvironment
from snake_game.snake_game_agent_version import Coordinate

environment = SnakeGymEnvironment()
environment.snake_game.GAME_SPEED = 1
environment.reset()
environment.render()


# Test eat food reward and sample 10 random actions
environment.snake_game.food = Coordinate((environment.snake_game.head.x + 10), environment.snake_game.head.y)
_, reward, _, info = environment.step(environment.snake_game.DIRECTION["right"])
print("Reward eating food:", reward)
print("Info:", info)
for action_number in range(1, 11):
    random_action = environment.action_space.sample()
    print(f"Action {action_number} Taken: {random_action}")
    observation, reward, _, _ = environment.step(random_action)
    print(f"Action {action_number} Reward: {reward}")
    print(f"Action {action_number} Observation: {observation}")


# Test suicide punishment
environment.reset()
head = environment.snake_game.head
GRID_SIZE = environment.snake_game.GRID_SIZE
environment.snake_game.body = [
    environment.snake_game.head,
    Coordinate(head.x - GRID_SIZE, head.y),
    Coordinate(head.x - GRID_SIZE * 2, head.y),
    Coordinate(head.x - GRID_SIZE * 3, head.y),
    Coordinate(head.x - GRID_SIZE * 4, head.y),
]
environment.step(environment.snake_game.DIRECTION["down"])
environment.step(environment.snake_game.DIRECTION["left"])
_, reward, _, _ = environment.step(environment.snake_game.DIRECTION["up"])
print("Reward agent suicide", reward)
environment.close()
