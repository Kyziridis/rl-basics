import Arena
from MCTS import MCTSAgent
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
from tictactoe.TicTacToeLogic import Board
import numpy as np
from utils import *
from time import time
from joblib import Parallel, delayed
import multiprocessing


# iters = np.linspace(500, 2000, 7, dtype=np.int16)
# data = []
# global_start = time()
# for game in range(3,5):
#     g = TicTacToeGame(game)
#     for i in iters:
#         for c in range(2):
#             np.random.seed(1000)
#             rp = RandomPlayer(g).play
#             mcts = MCTSAgent(g, iters=i, c=c, rollout_iter=1).play
#             start = time()
#             arena_rp_hp = Arena.Arena(mcts, rp, g, display=display)
#             wins, loss, draw = arena_rp_hp.playGames(4, verbose=False)
#             time_total = time() - start
#             data.append([game, i, c, wins, loss, draw, time_total])
# print(data)
# x = np.array(data)
# print('Time: ' + str(time() - global_start))
# np.save('data_export_noP', x)

def experiment(game, iters):
    g = TicTacToeGame(game)
    for c in range(2):
        np.random.seed(1000)
        rp = RandomPlayer(g).play
        mcts = MCTSAgent(g, iters=iters, c=c, rollout_iter=1).play
        start = time()
        arena_rp_hp = Arena.Arena(mcts, rp, g, display=display)
        wins, loss, draw = arena_rp_hp.playGames(4, verbose=False)
        time_total = time() - start
        data.append([game, iters, c, wins, loss, draw, time_total])
    return data

print('Start Parallel')
global_start = time()
iters = np.linspace(500, 2000, 7, dtype=np.int16)
data = []
data = Parallel(n_jobs=4)(delayed(experiment)(game, i) for game in range(3,5) for i in iters)
np.save('data_export', data) # 5 lepta putanes
print('Time: ' + str(time() - global_start))
