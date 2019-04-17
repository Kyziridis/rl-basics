import Arena
from Qlearning import QAgent
#from connect4.Connect4Game import Connect4Game, display
#from connect4.Connect4Players import *
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *

import numpy as np
from utils import *
import time

n_episodes = [120000, 160000, 200000]
reps = 5
n_games = 100
#lrs = [0.01, 0.05, 0.1]
lrs = [0.01]
epsilon_config = ['f']
game=3
#def experiment(game):
np.random.seed(556)
g = TicTacToeGame(game)
if game == 3:
    total_episodes = n_episodes[0]
elif game == 4:
    total_episodes = n_episodes[1]
else:
    total_episodes = n_episodes[2]
ep_step = 2000
ep_range = np.arange(0, total_episodes + ep_step, ep_step) + 1
ep_range[0] = 0
ep_range = ep_range.astype(int)
ep_range = ep_range[0:-1]
for lr in lrs:
    for i in epsilon_config:
        test_wins = []
        if i == 'f':
            q_agent = QAgent(g, episodes=total_episodes, lr=lr, epsilon=0.2, dc=1, e_min=0.001)
            rp = RandomPlayer(g).play
            q_agent_play = q_agent.play
        else:
            q_agent = QAgent(g, episodes=total_episodes, lr=lr, epsilon=1, dc=0.99, e_min=0.001)
            rp = RandomPlayer(g).play
            q_agent_play = q_agent.play
        for idx, episode in enumerate(ep_range):
            if episode == ep_range[-1]:
                break
            if episode == 0:
                print('Training for Episodes ', 0, ' to ', ep_range[idx + 1] - 1, '...', sep='')
            elif episode == ep_range[-2]:
                print('Training for Episodes ', episode - 1, ' to ', total_episodes, '...', sep='')
            else:
                print('Training for Episodes ', episode - 1, ' to ', ep_range[idx + 1] - 1, '...', sep='')
            q_agent.train(cur_episode=episode)
            print('Training Finished.')
            print('Playing in Arena...')
            wins = 0
            for i in range(reps):
                arena_rp_op = Arena.Arena(q_agent_play, rp, g, display=display)
                w, _, _ = arena_rp_op.playGames(n_games, verbose=False)
                wins += w
            test_wins.append(wins / (reps * n_games))
            print('\n')
        np.save('train_wr_tictactoe_' + str(game) + '_' + str(lr) + '_' + str(i), q_agent.total_wins)
        np.save('train_ep_tictactoe_' + str(game) + '_' + str(lr) + '_' + str(i), q_agent.total_eps)
        np.save('test_wr_tictactoe_' + str(game) + '_' + str(lr) + '_' + str(i), test_wins)
        print('\n')

#n_episodes = [40000, 80000, 120000]
#games = [3, 4, 5]
#games = [3]
#game=3
#reps = 5
#n_games = 100
#lrs = [0.01, 0.05, 0.1]
#lrs = [0.01]
#epsilon_config = ['f']
#epsilon_config = ['f', 'd']
#for game in games:
