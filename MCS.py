import numpy as np

'''
write MCS player as a small exercise before you implement the MCTS
'''

class MCSAgent():
    def __init__(self, game, nSims, name):
        self.game = game
        self.nSims = nSims
        self.name = name

    def simulate(self, board):
        initialBoard = board
        curPlayer = 1
        qValues = []
        if self.name == 'TicTacToe': 
            qValues = [0] * (self.game.getActionSize() -1)
        elif self.name == 'Connect4':
            qValues = [0] * (self.game.getActionSize())

        valids = self.game.getValidMoves(board, 1)
        self.candidate_acts = np.where(valids == 1)[0]
        for a in self.candidate_acts:   
            for i in range(self.nSims):
                board = initialBoard    
                board, curPlayer = self.game.getNextState(board, 1, a)
                while self.game.getGameEnded(board, curPlayer)==0:
                    valid_acts = np.where(self.game.getValidMoves(board, 1) == 1)[0]
                    rand_act = np.random.choice(valid_acts)
                    board, curPlayer = self.game.getNextState(board, curPlayer, rand_act)
                qValues[a] += self.game.getGameEnded(board, 1)
        qValues = np.array(qValues) / self.nSims 
        return qValues

        
    def play(self, board):
        # Gamw to camelcase trava grapse java!
        qValues = self.simulate(board)
        qValues_valid = qValues[self.candidate_acts]
        l = []
        for i in range(len(qValues_valid)):
            l.append((qValues_valid[i], self.candidate_acts[i]))
        l.sort()
        action = l[-1][1]    
        return action
