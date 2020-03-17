import Arena
from Agents.Qlearning_experiments_op import QAgent
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
from utils import *
from time import time

"""
The script is for experiments on Qlearning with fix parameter tunning trained against OP.
We experiment only with two configurations of the Connect4Game games = [(4, 5, 3), (5, 6, 4)]
"""

def experiment(game):
    np.random.seed(556)
    height = game[0]
    width = game[1]
    win_streak = game[2]
    g = Connect4Game(height, width, win_streak)
    if game == (4, 5, 3):
        total_episodes = n_episodes[0]
        ep_step = 80000
        ep_range = np.arange(0, total_episodes + ep_step, ep_step) + 1
        ep_range[0] = 0
        ep_range = ep_range.astype(int)
    elif game == (5, 6, 4):
        total_episodes = n_episodes[1]
        ep_step = 160000
        ep_range = np.arange(0, total_episodes + ep_step, ep_step) + 1
        ep_range[0] = 0
        ep_range = ep_range.astype(int)
    else:
        pass
    for lr in lrs:
        for i in epsilon_config:
            print('Config: Game', game, 'lr', lr, 'epsilon', i)
            test_wr_list_op = []
            test_wr_list_rp = []
            test_wr_op = []
            test_wr_rp = []
            if i == 'f':
                q_agent = QAgent(g, episodes=total_episodes, lr=lr, epsilon=0.2, dc=1, e_min=0.001, ep_arena=ep_step)
                op = OneStepLookaheadConnect4Player(g, verbose=False).play
                rp = RandomPlayer(g).play
                q_agent_play = q_agent.play
            else:
                q_agent = QAgent(g, episodes=total_episodes, lr=lr, epsilon=1, dc=0.99, e_min=0.001, ep_arena=ep_step)
                op = OneStepLookaheadConnect4Player(g, verbose=False).play
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
                wins_op, wins_rp = 0, 0
                temp_op = []
                temp_rp = []
                for repet in range(reps):
                    arena_rp_op = Arena.Arena(q_agent_play, op, g, display=display)
                    w_op, _, _ = arena_rp_op.playGames(n_games, verbose=False)
                    temp_op.append(w_op / n_games) 
                    wins_op += w_op

                    arena_qp_rp = Arena.Arena(q_agent_play, rp, g, display=display)
                    w_rp, _, _ = arena_qp_rp.playGames(n_games, verbose=False)
                    temp_rp.append(w_rp / n_games)
                    wins_rp += w_rp

                test_wr_list_op.append(temp_op)
                test_wr_op.append(wins_op / (reps * n_games))
                test_wr_list_rp.append(temp_rp)
                test_wr_rp.append(wins_rp / (reps * n_games))
                print('\n')
            end = time()
            training_time = np.array([end - start])
            np.save('Qlearning_results/train_wr_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_op', q_agent.total_wins)
            np.save('Qlearning_results/train_ep_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_op', q_agent.total_eps)
            np.save('Qlearning_results/test_wr_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_op', test_wr_op)
            np.save('Qlearning_results/test_wr_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', test_wr_rp)
            np.save('Qlearning_results/test_wr_list_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_op', test_wr_list_op)
            np.save('Qlearning_results/test_wr_list_connect4_' + str(game) + '_' + str(lr) + '_' + str(i) + '_rp', test_wr_list_rp)
            np.save('Qlearning_results/training_time_' + str(game) + '_' + str(lr) + '_' + str(i) + '_op', training_time)
            print('\n')

n_episodes = [2000000, 4000000]
reps = 10
n_games = 100
lrs = [0.01]
games = [(4, 5, 3), (5, 6, 4)]
epsilon_config = ['f']
Parallel(n_jobs=2)(delayed(experiment)(g) for g in games)
