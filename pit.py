import Arena
from Agents.MCTS_time import MCTSAgent
from Agents.MCS_time import MCSAgent
from Agents.Qlearning import QAgent

# from connect4.Connect4Game import Connect4Game, display
# from connect4.Connect4Players import *
# from connect4.Connect4Logic import Board

from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
from tictactoe.TicTacToeLogic import Board

import numpy as np
from utils import *
from time import time

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
np.random.seed(66)
# g = Connect4Game(6,7,4)
g = TicTacToeGame(3)
#

# All players
rp = RandomPlayer(g).play
#

# Time is in microseconds
# Put big number of iterations in order to be stoped by the time
mcs = MCSAgent(g, nSims=1000000, time=300000).play
# Same here in mcts
mcts = MCTSAgent(g, iters=1000000, c=1, rollout_iter=100, time=10000000).play
#

# For fix epsilon-greedy put dc=1
# For dynamical epsilon-greedy put e.g. dc=0.99
ql = QAgent(game=g, episodes=4500 , lr=0.1, epsilon=0.2, dc=1, e_min=0.001)
# The Q-Agent will be trained first and then play in arena
ql.train()
# Use the ql_player in the Arena
ql_player = ql.play
#

#One = OneStepLookaheadConnect4Player(g).play
#
#hp = HumanConnect4Player(g).play
#hp = HumanTicTacToePlayer(g).play
#
# op=OneStepLookaheadConnect4Player(g).play
#

###### PLAY IN ARENA ###############
start = time()
arena_rp_hp = Arena.Arena(ql_player, mcts, g, display=display)
print('')
print(arena_rp_hp.playGames(6, verbose=False))
print('Time Elapsed: ' + str(round(time()-start)) + ' secs')