'''
write MCTS player by yourself
'''
import numpy as np
import sys
from datetime import datetime

class MCTSAgent():
    def __init__(self, game, iters, c, rollout_iter, time):
        self.game = game
        self.S = {}
        self.iters = iters
        self.c = c
        self.state_stats = {}
        self.history = []
        self.rollout_iter = rollout_iter
        self.time = time

    def UCB(self, q, N, n, epsilon=1e-8):
        return q + self.c* np.sqrt(np.clip(np.log(N + epsilon), a_min=0, a_max=np.Inf) / (n + epsilon))

    def rollout(self, board, curPlayer):
        Q = []
        initBoard = board
        C = curPlayer
        rollout_player = -C
        for i in range(self.rollout_iter):
            board = initBoard
            valids = self.game.getValidMoves(board, 1)
            act = np.random.randint(self.game.getActionSize())
            while valids[act] !=1:
                act = np.random.randint(self.game.getActionSize())
            board, curPlayer = self.game.getNextState(board, C, act)
            while self.game.getGameEnded(board, curPlayer)==0:
                valids = self.game.getValidMoves(board, 1)
                act = np.random.randint(self.game.getActionSize())
                while valids[act] !=1:
                    act = np.random.randint(self.game.getActionSize())
                board, curPlayer = self.game.getNextState(board, curPlayer, act)
            Q.append(self.game.getGameEnded(board, rollout_player))
        return np.mean(Q)

    def selection(self, board, curPlayer):
        s = board.tostring()
        valids = self.game.getValidMoves(board, curPlayer)
        possible_acts = np.arange(self.game.getActionSize())
        valid_acts = possible_acts[valids==1]
        neg_acts = possible_acts[valids==0]
        ucb_values = np.zeros(self.game.getActionSize())
        ucb_values[neg_acts] = -1e+11
        for a in valid_acts:
            next_s, _ = self.game.getNextState(board, curPlayer, a)
            next_s = next_s.tostring()
            ucb_values[a] = self.UCB(self.state_stats[next_s][1], self.state_stats[s][0], self.state_stats[next_s][0])
        max_ = max(ucb_values)
        max_indx = np.where(ucb_values == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(max_indx)
            return action
        action = np.argmax(ucb_values)
        return action

    def expand(self, board, curPlayer):
        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(board, curPlayer)
        valid_acts = possible_acts[valids==1]
        for act in valid_acts:
            next_s, _ = self.game.getNextState(board, curPlayer, act)
            next_s = next_s.tostring()
            if next_s not in self.state_stats:
                self.state_stats[next_s] = np.array([0.,0.])

    def update(self, board):
        for idx, s in enumerate(self.history):
            self.state_stats[s][0] += 1
            if idx % 2 == 0:
                q = self.game.getGameEnded(board, -1)
                self.state_stats[s][1] = self.state_stats[s][1] + (q - self.state_stats[s][1]) / self.state_stats[s][0]
            else:
                q = self.game.getGameEnded(board, 1)
                self.state_stats[s][1] = self.state_stats[s][1] + (q - self.state_stats[s][1]) / self.state_stats[s][0]

    def simulate(self, board, curPlayer):
        

        c = datetime.now() - self.start
        self.microsecs = c.seconds * 1e6 + c.microseconds
        if self.microsecs >= self.time:
            return
 
        s = board.tostring()
        self.history.append(s)

        # Check if the current state is visited and call rollout
        if s not in self.S:
            self.S[s] = 1

            if self.game.getGameEnded(board, curPlayer) != 0:
                self.update(board)
                return

            if self.sim == 0:
                self.state_stats[s] = np.array([0.,0.])
                curPlayer = 1
                return

            q = self.rollout(board, curPlayer)
            self.update(board)
            return

        if self.game.getGameEnded(board, curPlayer) != 0:
            return

        # Take valid actions
        self.expand(board, curPlayer)
        chosen_action = self.selection(board, curPlayer)

        # Traverse
        next_s, curPlayer = self.game.getNextState(board, curPlayer, chosen_action)
         
        # Recursion
        self.simulate(next_s, curPlayer)


    def play(self, board):
        self.S={}
        self.state_stats = {}
        curPlayer = 1
        self.start = datetime.now()
        for self.sim in range(self.iters):
            self.history = []
            self.simulate(board, curPlayer)
            if self.microsecs >= self.time:
                break

        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        valid_acts = possible_acts[valids==1]

        if len(valid_acts) == 1:
            return valid_acts[0]
        else:
            candidates = list(self.state_stats.values())[1:len(valid_acts)+1]
            candidates = np.array(list(map(lambda x: x[1], candidates)))
            max_indx = np.where(candidates == max(candidates))[0]
            if len(max_indx) != 1:
                action = np.random.choice(valid_acts[max_indx])
                return action

            indx_max = candidates.argmax()
            action = valid_acts[indx_max]
            return action
