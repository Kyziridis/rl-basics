import numpy as np
from time import time
from datetime import datetime


'''
This script is the MCS agent class. It takes a state from the Arena and 
it plays many simulations in order to select the optimal action.
It contains response time contrains and it is used for the experiments.
'''     

class MCSAgent():
    def __init__(self, game, nSims, time):
        self.game = game
        self.time = time
        self.nSims = nSims

    def simulate(self, board):
        initialBoard = board
        curPlayer = 1
        qValues = []
        qValues = [0] * (self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        self.candidate_acts = np.where(valids == 1)[0]
        n_acts = np.ones(self.game.getActionSize(), dtype=np.float32)
        start = datetime.now()
        for sim in range(self.nSims):
            for a in self.candidate_acts:
                if sim > 0:
                    n_acts[a] += 1
                board = initialBoard
                board, curPlayer = self.game.getNextState(board, 1, a)
                while self.game.getGameEnded(board, curPlayer)==0:
                    act = np.random.randint(self.game.getActionSize())
                    valids = self.game.getValidMoves(board,1)
                    while valids[act] !=1:
                        act = np.random.randint(self.game.getActionSize())
                    board, curPlayer = self.game.getNextState(board, curPlayer, act)
                qValues[a] += self.game.getGameEnded(board, 1)
                c = datetime.now() - start
                microsecs = c.seconds * 1e6 + c.microseconds
                if microsecs >= self.time:
                    break
            if microsecs >= self.time:
                break
        qValues = np.array(qValues) / n_acts
        return qValues

    def play(self, board):
        qValues = self.simulate(board)
        qValues_valid = qValues[self.candidate_acts]

        max_ = max(qValues_valid)
        max_indx = np.where(qValues_valid == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(self.candidate_acts[max_indx])
            return action

        indx = qValues_valid.argmax()
        action = self.candidate_acts[indx]
        return action
