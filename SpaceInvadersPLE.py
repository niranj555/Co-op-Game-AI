from ple.games.SpaceInvadersGame import SpaceInvadersGame
from ple import PLE


game = SpaceInvadersGame()
p = PLE(game, fps=30, display_screen=True)
#agent = myAgentHere()

p.init()
reward = 0.0

for i in range(100):
   if p.game_over():
           p.reset_game()

   observation = p.getScreenRGB()
   #action = agent.pickAction(reward, observation)
   allowed_actions=p.getActionSet()
   reward = p.act(allowed_actions[4])