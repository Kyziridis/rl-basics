import Arena
from DEMO.MCS import MCSAgent
from DEMO.MCTS import MCTSAgent
from DEMO.Qlearning import QAgent

from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
from tictactoe.TicTacToeLogic import Board

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
g = TicTacToeGame(3)

# all players
rp = RandomPlayer(g).play

mcs = MCSAgent(g,nSims=500).play

mcts = MCTSAgent(g, iters=20, c=3, rollout_iter=1).play

Q = QAgent(g, episodes=25000, lr=0.1, epsilon=0.2, dc=1, e_min=0.001)

Qplayer = Q.play

hp = HumanTicTacToePlayer(g).play


print("This is a Demo with all the three agents playing four TicTacToe games in Arena\n")
flag = input('Do you want to play in Human Mode ?  >_  [y/n]  ')
if flag == 'y' or flag=='Y' or flag=='yes' or flag=='YES':

    opponent = input('Select Opponent [mcs/mcts/Ql]  >_  ')

    if opponent == 'mcs':
        arena_rp_hp = Arena.Arena(hp, mcs, g, display=display)
        print(arena_rp_hp.playGames(2, verbose=True))
    elif opponent == 'mcts':
        arena_rp_hp = Arena.Arena(hp, mcts, g, display=display)
        print(arena_rp_hp.playGames(2, verbose=True))
    elif opponent =='Ql':
        Q.train()
        arena_rp_hp = Arena.Arena(hp, Qplayer, g, display=display)
        print(arena_rp_hp.playGames(2, verbose=True))

else:

    print("\nMCS Vs Random Player")
    arena_rp_hp = Arena.Arena(mcs, rp, g, display=display)
    print(arena_rp_hp.playGames(4, verbose=False))
    print('')
    print('MCTS Vs Random Player')
    arena_rp_hp = Arena.Arena(mcts, rp, g, display=display)
    print(arena_rp_hp.playGames(4, verbose=False))
    print('')
    print('Q-Agent Vs Random Player')
    Q.train()
    print('')
    arena_rp_hp = Arena.Arena(Qplayer, rp, g, display=display)
    print(arena_rp_hp.playGames(1000, verbose=False))

