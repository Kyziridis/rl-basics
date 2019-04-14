import Arena
from MCS import MCSAgent
from connect4.Connect4Game import TicTacToeGame, display
from connect4.Connect4Players import *

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from utils import *
import time

#
# global_start = time.time()
# data = []
# for game in range(3, 6):
#     g = TicTacToeGame(game)
#     for i in iters:
#         for seed in seeds:
#             np.random.seed(seed)
#             rp = RandomPlayer(g).play
#             mcs = MCSAgent(g, i).play
#             start = time.time()
#             arena_rp_op = Arena.Arena(mcs, rp, g, display=display)
#             wins, losses, draws = arena_rp_op.playGames(100, verbose=False)
#             time_total = time.time() - start
#             data.append([game, i, seed, wins, losses, draws, time_total])
#
# np.save('data_export_noP', data)
# print('Time: ' + str(time.time() - global_start))

##################################
##################################

def experiment(game, iters):
    height, width, win_streak = game
    g = Connect4(height, width, win_streak)
    for seed in seeds:
        np.random.seed(seed)
        rp = RandomPlayer(g).play
        mcs = MCSAgent(g, iters).play
        start = time.time()
        arena_rp_op = Arena.Arena(mcs, rp, g, display=display)
        wins, losses, draws = arena_rp_op.playGames(100, verbose=False)
        time_total = time.time() - start
        data.append([game, iters, seed, wins, losses, draws, time_total])
    return data

seeds = 1000*(np.arange(5) + 1)
iters = np.linspace(250, 2000, 8, dtype=np.int16)
games = [(4, 5, 3), (5, 6, 7), (6, 7, 4)]
global_start = time.time()
data = []
data = Parallel(n_jobs=8)(delayed(experiment)(game, i) for game in games for i in iters)
np.save('mcs_tictactoe_results', data)
print('Time: ' + str(time.time() - global_start))
