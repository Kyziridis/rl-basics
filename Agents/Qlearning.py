'''
write a simple Q-learning player
The current script is the Q-Agent. 
It is only for pit.py we did NOT use for the experiments 
You can use it in pit.py
'''
import numpy as np
import sys
from time import time
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class END(Exception): pass

class QAgent():
    def __init__(self, game, episodes, lr, epsilon, dc, e_min):
        self.Q = {}
        self.game = game
        self.episodes = episodes
        #self.ep = 0
        self.lr = lr
        self.epsilon = []
        self.flag = True
        self.wins = 0
        self.draw = 0
        self.loss = 0
        self.gamma = 0.99
        self.tau = 0.01
        self.e = epsilon
        self.dc = dc
        self.e_ = e_min
        self.config = {'epsilon:':self.e,
                        'discount_e:':self.dc,
                        'epsilon_min:':self.e_,
                        'gamma:':self.gamma,
                        'learning rate:': self.lr}
        self.total_eps = []
        self.total_wins = []

    def update(self, R, Q_prime, s):
        Q_new = self.Q[s] + self.lr*(R + self.gamma*Q_prime - self.Q[s])
        self.Q[s] = Q_new

    def e_greedy(self, board, actions_q):
        if np.random.rand() <= self.e :
            valid_acts = np.where(actions_q!=-1e+9)[0]
            action = np.random.choice(valid_acts)
            self.epsilon.append(self.e)
            if self.e > self.e_:
                self.e *= self.dc
            return action
        max_ = max(actions_q)
        max_indx = np.where(actions_q == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(max_indx)
            return action
        action = np.argmax(actions_q)
        return action

    def init_q(self, board):
        temp = []
        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        valid_acts = possible_acts[valids==1]
        neg_acts = possible_acts[valids==0]
        actions_q = np.zeros_like(possible_acts, dtype=np.float32)
        actions_q[neg_acts] = -1e+9
        for action in valid_acts:
            next_s, _ = self.game.getNextState(board, 1, action)
            s_next = next_s.tostring()
            temp.append(s_next)
            if s_next not in self.Q:
                self.Q[s_next] = 0.
            actions_q[action] = self.Q[s_next]
        return temp, actions_q


    def opponent_play(self, board, curPlayer):
        act = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board,1)
        while valids[act] !=1:
            act = np.random.randint(self.game.getActionSize())
        board, curPlayer = self.game.getNextState(board, curPlayer, act)
        return board, curPlayer


    def check_terminal(self, board, s):
        if self.game.getGameEnded(board, 1) != 0:
            ret = self.game.getGameEnded(board, 1)
            Q_prime = 0
            self.update(ret, Q_prime, s)
            if ret == 1:
                self.wins += 1
            elif ret==-1:
                self.loss += 1
            else:
                self.draw += 1
            end = time()
            raise END
        else:
            pass


    def max_q(self, temp):
        qs = []
        for state in temp:
            qs.append(self.Q[state])
        Q_prime = max(qs)
        ret = 0
        return ret, Q_prime


    def simulate(self):
        init_board = self.game.getInitBoard()
        s = init_board.tostring()
        temp = []
        for self.ep in tqdm(range(self.cur_episode, self.episodes + 1)):
            if self.ep == int(np.round(self.episodes * (2/3))) and self.dc == 1:
                self.e = 0
            board = init_board
            # Init the first episode
            if self.ep == 0:
                self.Q[s] = 0.

            # Check the players
            if self.ep % 2 == 0:
                curPlayer = 1
            else:
                curPlayer = -1
                board, curPlayer = self.opponent_play(board, curPlayer)

            # Expand
            temp, actions_q = self.init_q(board)
            self.start = time()
            try:
                while self.game.getGameEnded(board, 1) == 0:

                    # Choose action with e_greedy policy
                    action = self.e_greedy(board, actions_q)

                    # My player exerts action!
                    board, curPlayer = self.game.getNextState(board, curPlayer, action)
                    s = board.tostring()
                    # check if my action resulted in terminal state
                    # If board == Terminal then update and break
                    self.check_terminal(board, s)

                    # Opponent exerts random action!
                    board, curPlayer = self.opponent_play(board, curPlayer)
                    # check if opponent's action resulted in terminal state
                    # If board == Terminal then update and break
                    self.check_terminal(board, s)

                    # Expand
                    temp, actions_q = self.init_q(board)
                    # Calculate Max_Q for Q_prime
                    ret, Q_prime = self.max_q(temp)
                    # Update
                    self.update(ret, Q_prime, s)

            except END:
                pass

    def train(self, cur_episode=0):
        self.cur_episode = cur_episode
        general_time = time()
        print('Train Q-Agent for %s episodes >_' %self.episodes)
        self.simulate()
        print('Training finished.')
        #print('Q:\n' + str(self.Q))
        print('Wins: %s | Loss: %s | Draw: %s |' %(self.wins, self.loss, self.draw)\
         + ' Total Training Time: ' + str(round(time()-general_time)) + ' secs'  )
        print('\nConfiguration: ', self.config)
        

    def play(self, board):
        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        valid_acts = possible_acts[valids==1]
        neg_acts = possible_acts[valids==0]
        actions_q = np.zeros_like(possible_acts, dtype=np.float32)
        actions_q[neg_acts] = -1e+9

        for action in valid_acts:
            next_s, _ = self.game.getNextState(board, 1, action)
            s_next = next_s.tostring()
            if s_next not in self.Q: self.Q[s_next] = 0.
            actions_q[action] = self.Q[s_next]

        max_ = max(actions_q)
        max_indx = np.where(actions_q == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(max_indx)
            return action
        final_action = np.argmax(actions_q)
        return final_action
