import Arena
from Agents.MCS_time import MCSAgent
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
        mcs = MCSAgent(g, nSims=100000000, time=m).play
        arena_rp_hp = Arena.Arena(mcs, rp, g, display=display)
        wins, loss, draw = arena_rp_hp.playGames(100, verbose=False)
        data.append([m, rep, wins, loss, draw])
    return data

print('Start Parallel')
global_start = time()
microsecs = np.array([5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000])
games = [3, 4, 5]
for i in games:
    global_start = time()
    g = TicTacToeGame(i)
    data = []
    data = Parallel(n_jobs=11)(delayed(experiment)(m) for m in microsecs)
    np.save('tictactoe_results_' + str(i), data) 
    print('Game: ' + str(i) +' Time: ' + str(time() - global_start))
