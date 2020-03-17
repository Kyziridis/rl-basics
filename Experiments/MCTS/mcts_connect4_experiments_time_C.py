import Arena
from Agents.MCTS_time import MCTSAgent
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.Connect4Logic import Board
import numpy as np
from utils import *
from time import time
from joblib import Parallel, delayed
import multiprocessing

"""
The script is for the MCTS experiments for the Connect4Game for different C.
Experimentation on 10 different microseconds for response time and 5 different C values.
On 3 different configurations of Connect4Game :games = [(4, 5, 3), (5, 6, 4), (6, 7, 4)].
Against both Random and OneStepLookahead opponent.
"""
def experiment(m):
    for c in range(6):
        mcts = MCTSAgent(g, iters=100000000, c=c, rollout_iter=1, time=m).play
        if player == 'rp':
            opponent = RandomPlayer(g).play
        else:
            opponent = OneStepLookaheadConnect4Player(g, verbose=False).play
        arena_rp_hp = Arena.Arena(mcts, opponent, g, display=display)
        wins, loss, draw = arena_rp_hp.playGames(100, verbose=False)
        data.append([m, c, wins, loss, draw])
    return data

print('Start Parallel Simulation for Connect4: (4,5,3) (5,6,4) (6,7,4)')
global_start = time()
microsecs = np.array([10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000])
games = [(4,5,3), (5,6,4), (6,7,4)]
players = ['rp', 'op']

for player in players:
    for i in games:
        global_start = time()
        g = Connect4Game(i[0], i[1], i[2])
        data = []
        data = Parallel(n_jobs=10)(delayed(experiment)(m) for m in microsecs)
        np.save('connect4_results_'+player+'_'+str(i), data) 
        print('Game: ' + str(i) + 'Opponent: '+player +' Time: ' + str(time() - global_start))
