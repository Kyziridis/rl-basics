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

"""
The script is for the MCTS experiments for the TicTacToe4Game regarding different C.
Experimentation on 5 different exploration values C for 10 different response times.
On 3 different configurations of TicTacToe :games = [(3x3), (4x4), (5x5)].
Against both Random opponent.
"""

def experiment(m):
    for c in range(6):
        rp = RandomPlayer(g).play
        mcts = MCTSAgent(g, iters=100000000, c=c, rollout_iter=1, time=m).play
        arena_rp_hp = Arena.Arena(mcts, rp, g, display=display)
        wins, loss, draw = arena_rp_hp.playGames(100, verbose=False)
        data.append([m, c, wins, loss, draw])
    return data

print('Start Parallel Simulation for TicTacToe: 4, 5, 6')
global_start = time()
microsecs = np.array([5000, 10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000])
games = [3,4,5]
for i in games:
    global_start = time()
    g = TicTacToeGame(i)
    data = []
    data = Parallel(n_jobs=11)(delayed(experiment)(m) for m in microsecs)
    np.save('tictacttoe_results_'+str(i), data) 
    print('Game: ' + str(i) +' Time: ' + str(time() - global_start))
