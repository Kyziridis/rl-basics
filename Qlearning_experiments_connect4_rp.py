import Arena
from Qlearning import QAgent
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
#from tictactoe.TicTacToeGame import TicTacToeGame, display
#from tictactoe.TicTacToePlayers import *
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
from utils import *
from time import time

def experiment(game):
    np.random.seed(556)
    height = game[0]
    width = game[1]
    win_streak = game[2]
    g = Connect4Game(height, width, win_streak)
    if game == (4, 5, 3):
        total_episodes = n_episodes[0]
    elif game == (5, 6, 4):
        total_episodes = n_episodes[1]
    else:
        total_episodes = n_episodes[2]
    ep_step = 2000
    ep_range = np.arange(0, total_episodes + ep_step, ep_step) + 1
    ep_range[0] = 0
    ep_range = ep_range.astype(int)
    for lr in lrs:
        for i in epsilon_config:
            print('Config: Game', game, 'lr', lr, 'epsilon', i)
            test_wr_list = []
            test_wr = []
            if i == 'f':
                q_agent = QAgent(g, episodes=total_episodes, lr=lr, epsilon=0.2, dc=1, e_min=0.001)
                rp = RandomPlayer(g).play
                q_agent_play = q_agent.play
            else:
                q_agent = QAgent(g, episodes=total_episodes, lr=lr, epsilon=1, dc=0.99, e_min=0.001)
                rp = RandomPlayer(g).play
                q_agent_play = q_agent.play
            start = time()
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
                temp = []
                for repet in range(reps):
                    arena_rp_op = Arena.Arena(q_agent_play, rp, g, display=display)
                    w, _, _ = arena_rp_op.playGames(n_games, verbose=False)
                    temp.append(w / n_games) 
                    wins += w
                test_wr_list.append(temp)
                test_wr.append(wins / (reps * n_games))
                print('\n')
            end = time()
            training_time = np.array([end - start])
            np.save('Qlearning_results/train_wr_connect4__' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', q_agent.total_wins)
            np.save('Qlearning_results/train_ep_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', q_agent.total_eps)
            np.save('Qlearning_results/test_wr_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', test_wr)
            np.save('Qlearning_results/test_wr_list_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', test_wr_list)
            np.save('Qlearning_results/training_time_' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', training_time)
            print('\n')

n_episodes = [120000, 160000, 200000]
reps = 5
n_games = 100
lrs = [0.01, 0.05, 0.1]
games = [(4, 5, 3), (5, 6, 4), (6, 7, 4)]
epsilon_config = ['f', 'd']
Parallel(n_jobs=3)(delayed(experiment)(g) for g in games)
