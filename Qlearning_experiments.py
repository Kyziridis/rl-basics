import Arena
from Qlearning import QAgent
#from connect4.Connect4Game import Connect4Game, display
#from connect4.Connect4Players import *
from tictactoe.TicTacToeGame import TicTacToeGame, display
from tictactoe.TicTacToePlayers import *

import numpy as np
from utils import *
import time

total_episodes = np.arange(0, 502000, 2000)
games = [3, 4, 5]
reps = 5
n_games = 100
for game in games:
    np.random.seed(556)
    g = TicTacToeGame(game)

    test_wins = []
    q_agent = QAgent(g, episodes=total_episodes[1], lr=0.1, epsilon=0.2, dc=1, e_min=0.001)
    rp = RandomPlayer(g).play
    q_agent_play = q_agent.play
    for idx, ep in enumerate(total_episodes):
        if ep == total_episodes[-1]:
            break
        q_agent.episodes = total_episodes[idx + 1]
        print('TicTacToe Game:', str(game))
        print('Training for', ep, 'Episodes...')
        q_agent.train(cur_episode=ep)
        print('Training Finished.')
        print('Playing in Arena...')
        wins = 0
        for i in range(reps):
            arena_rp_op = Arena.Arena(q_agent_play, rp, g, display=display)
            w, _, _ = arena_rp_op.playGames(n_games, verbose=False)
            wins += w
        test_wins.append(wins / (reps * n_games))
        print('\n')
    #print(np.array(q_agent.total_wins) / np.array(q_agent.total_eps))
    #print(test_wins)
    np.save('train_wr_tictactoe' + str(g) + '.npy', q_agent.total_wins)
    np.save('train_ep_tictactoe' + str(g) + '.npy', q_agent.total_eps)
    np.save('test_wr_tictactoe' + str(g) + '.npy', test_wins)
