import Arena
from Agents.MCTS_time import MCTSAgent
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *
from tictactoe.TicTacToeLogic import Board
import numpy as np
from utils import *
from time import time
from joblib import Parallel, delayed
import multiprocessing


def experiment(m):
    for rep in range(5):
        rp = RandomPlayer(g).play
        mcts = MCTSAgent(g, iters=100000000,c=c_best, rollout_iter=1, time=m).play
        arena_rp_hp = Arena.Arena(mcts, rp, g, display=display)
        wins, loss, draw = arena_rp_hp.playGames(100, verbose=False)
        data.append([m, rep, wins, loss, draw])
    return data

print('Start Parallel')
global_start = time()
microsecs = np.array([5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000])
games = [3, 4, 5]
c = [2, 5 ,1]
for i,cs in zip(games,c):
    c_best = cs
    global_start = time()
    g = TicTacToeGame(i)
    data = []
    data = Parallel(n_jobs=11)(delayed(experiment)(m) for m in microsecs)
    np.save('mcts_best_tictactoe_results_' + str(i), data) 
    print('Game: ' + str(i) +' Time: ' + str(time() - global_start))
