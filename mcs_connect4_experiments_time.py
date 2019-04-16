import Arena
from MCS_time import MCSAgent
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.Connect4Logic import Board
import numpy as np
from utils import *
from time import time
from joblib import Parallel, delayed
import multiprocessing


def experiment(m):
    for rep in range(5):
        if player == 'rp':
            pl = RandomPlayer(g).play
        else:
            pl = OneStepLookaheadConnect4Player(g, verbose=False).play
        mcs = MCSAgent(g, nSims=100000000, time=m).play
        arena_rp_hp = Arena.Arena(mcs, pl, g, display=display)
        wins, loss, draw = arena_rp_hp.playGames(100, verbose=False)
        data.append([m, rep, wins, loss, draw])
    return data

print('Start Parallel')
global_start = time()
microsecs = np.array([10000, 50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000])
games = [(4, 5, 3), (5, 6, 4), (6, 7, 4)]
players = ['rp', 'op']
for player in players:
    for i in games:
        height = i[0]
        width = i[1]
        win_streak = i[2]
        global_start = time()
        g = Connect4Game(height, width, win_streak)
        data = []
        data = Parallel(n_jobs=10)(delayed(experiment)(m) for m in microsecs)
        np.save('connect4_results_' + player + '_' + str(height) + str(width) + str(win_streak), data) 
        print('Game: ' + str(i) + ' against ' + player + ', Time: ' + str(time() - global_start))
