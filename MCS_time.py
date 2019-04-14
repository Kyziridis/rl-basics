import numpy as np

'''
write MCS player as a small exercise before you implement the MCTS
'''

class MCSAgent():
    def __init__(self, game, nSims):
        self.game = game
        self.nSims = nSims

    def simulate(self, board):
        initialBoard = board
        #
        curPlayer = 1
        qValues = []
        qValues = [0] * (self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        self.candidate_acts = np.where(valids == 1)[0]
        for a in self.candidate_acts:
            for i in range(self.nSims):
                board = initialBoard
                board, curPlayer = self.game.getNextState(board, 1, a)
                while self.game.getGameEnded(board, curPlayer)==0:
                    # valid_acts = np.where(self.game.getValidMoves(board, 1) == 1)[0]
                    # rand_act = np.random.choice(valid_acts)
                    act = np.random.randint(self.game.getActionSize())
                    valids = self.game.getValidMoves(board,1)
                    while valids[act] !=1:
                        act = np.random.randint(self.game.getActionSize())
                    board, curPlayer = self.game.getNextState(board, curPlayer, act)
                qValues[a] += self.game.getGameEnded(board, 1)
        qValues = np.array(qValues) / self.nSims
        return qValues


    def play(self, board):
        qValues = self.simulate(board)
        qValues_valid = qValues[self.candidate_acts]
        # print('qValues: ', qValues)
        # print('Candi ', self.candidate_acts)

        max_ = max(qValues_valid)
        max_indx = np.where(qValues_valid == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(self.candidate_acts[max_indx])
            # print('Random act: ', action)
            return action

        indx = qValues_valid.argmax()
        action = self.candidate_acts[indx]
        # print('Kanoniko action ', action)
        return action

        # l = []
        # for i in range(len(qValues_valid)):
        #     l.append((qValues_valid[i], self.candidate_acts[i]))
        # l.sort()
        # action = l[-1][1]
        # return action
