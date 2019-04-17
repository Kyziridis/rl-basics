import Arena
from Qlearning import QAgent
#from connect4.Connect4Game import Connect4Game, display
#from connect4.Connect4Players import *
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *

import numpy as np
from utils import *
import time

g = TicTacToeGame(3)

np.random.seed(556)
# all players
#lrs = [0.001, 0.01, 0.05, 0.1, 1] # Kaname arxiko search, prokeimenou na vroume optimal range. TO DO: Search anamesa se 0.005 mexri 0.1.
#lrs = [0.005, 0.008, 0.01, 0.02, 0.05, 0.07, 0.1]
lrs = [0.01]
#ep_range = np.linspace(0, 40000, 9) + 1
total_episodes = np.arange(2, 11) * 10000
wr_episodes = []
for episodes in total_episodes:
    ep_range = np.linspace(0, episodes, 9) + 1
    ep_range[0] = 0
    ep_range = ep_range.astype(int)
    ep_range = ep_range[0:8]

    rp = RandomPlayer(g).play
    q_agent = QAgent(g, episodes=episodes, lr=0.01, epsilon=0.2, dc=1, e_min=0.001)
    q_agent_play = q_agent.play

    print('========')
    print('Total Episodes:', episodes)
    print('========')
    print('\n')
    arena_wins = []
    reps = 5
    for idx, ep in enumerate(ep_range):
        print('Starting training for episode:', ep)
        q_agent.train(cur_episode=ep)
        print('Playing in Arena.')
        print('\n')
        wins = 0
        for i in range(reps):
            arena_rp_op = Arena.Arena(q_agent_play, rp, g, display=display)
            w, _, _ = arena_rp_op.playGames(100, verbose=False)
            wins += w
        arena_wins.append(wins / (reps * 100))
    wr_episodes.append(arena_wins)
#np.save('wr_episodes4.npy', wr_episodes)
#print(mean_aw)
    #np.save('total_wins_' + str(lr) + '.npy', q_agent.total_wins)
    #np.save('total_eps_' + str(lr) + '.npy', q_agent.total_eps)
    #np.save('arena_wins_' + str(lr) + '.npy', arena_wins)
#op=OneStepLookaheadConnect4Player(g).play

#start = time.time()
#arena_rp_op = Arena.Arena(q_agent_play, rp, g, display=display)
#print(arena_rp_op.playGames(40, verbose=False))
# a, b, c = arena_rp_op.playGames(100, verbose=False)
# t = (a,b,c)
# print(t)
# end = time.time()
# print("Time:", str(end - start))
