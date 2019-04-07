'''
write a simple Q-learning player
'''
import numpy as np
import sys
from time import time
import matplotlib.pyplot as plt
import pickle

class END(Exception): pass

class QAgent():
    def __init__(self, game, episodes, lr):
        self.Q = {}
        self.game = game
        self.episodes = episodes
        self.lr = lr
        self.epsilon = []
        self.flag = True
        self.wins = 0
        self.draw = 0
        self.loss = 0
        self.gamma = 0.99
        self.tau = 0.01
        self.e = 1
        self.dc = 0.99
        self.e_ = 0.01

    def update(self, R, Q_prime, s):
        Q_new = self.Q[s] + self.lr*(R + self.gamma*Q_prime - self.Q[s])
        self.Q[s] = Q_new

    def e_greedy(self, board, actions_q):
        if np.random.rand() <= self.e :
            valid_acts = np.where(actions_q!=-1e+9)[0]
            action = np.random.choice(valid_acts)
            if self.e > self.e_:
                self.e *= self.dc
            #print('Epsilon Action: ', action)
            return action
        #print('Actions_q: '+ str(actions_q))
        max_ = max(actions_q)
        max_indx = np.where(actions_q == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(max_indx)
            return action
        action = np.argmax(actions_q)
        #print('Policy Action: ', action)
        self.epsilon.append(self.e)
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
            #s_next = str(next_s.flatten())
            s_next = next_s.tostring()
            temp.append(s_next)
            if s_next not in self.Q:
                self.Q[s_next] = 0.
            #print('Q[new_state]: ', self.Q[s_next])
            actions_q[action] = self.Q[s_next]
        return temp, actions_q


    def opponent_play(self, board, curPlayer):
        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(board, curPlayer)
        valid_acts = possible_acts[valids==1]
        action = np.random.choice(valid_acts)
        #print('Action diko tou: ', action)
        board, curPlayer = self.game.getNextState(board, curPlayer, action)
        return board, curPlayer


    def check_terminal(self, board, s):
        if self.game.getGameEnded(board, 1) != 0:
            ret = self.game.getGameEnded(board, 1)
            Q_prime = 0
            self.update(ret, Q_prime, s)
            # if self.ep % 500 == 0:
            #     if 0.0 not in self.Q.values():
            #         self.flag = False
            #         print('flag', self.flag)
            if ret == 1:
                self.wins += 1
            elif ret==-1:
                self.loss += 1
            else:
                self.draw += 1
            end = time()
            #print("\nEpisode: %s  |  Reward: %s " %(self.ep, ret))
            # self.ep += 1
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
        #s = str(init_board.flatten())
        temp = []
        for self.ep in range(self.episodes):
            board = init_board
            if self.ep % 2 == 0:
                curPlayer = 1
            else:
                curPlayer = -1
            #curPlayer = 1
            if self.ep == 0:
                self.Q[s] = 0.

            if curPlayer == -1:
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
                    #s = str(board.flatten())
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


    def train(self):
        general_time = time()
        self.simulate()
        #print('Q:\n' + str(self.Q))
        #print('Wins: %s | Loss: %s | Draw: %s |' %(self.wins, self.loss, self.draw)\
        # + ' Total Training Time: ' + str(round(time()-general_time))  )
        #plt.figure()
        #plt.plot(self.epsilon)
        #plt.xlabel('Iterations')
        #plt.ylabel('Epsilon')
        #plt.show()
        with open('Q_table.pickle', 'wb') as handle:
            pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def play(self, board):
        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        valid_acts = possible_acts[valids==1]
        neg_acts = possible_acts[valids==0]
        actions_q = np.zeros_like(possible_acts, dtype=np.float32)
        actions_q[neg_acts] = -1e+9

        for action in valid_acts:
            next_s, _ = self.game.getNextState(board, 1, action)
            #s_next = str(next_s.flatten())
            s_next = next_s.tostring()
            actions_q[action] = self.Q[s_next]

        max_ = max(actions_q)
        max_indx = np.where(actions_q == max_)[0]
        if len(max_indx) != 1:
            action = np.random.choice(max_indx)
            return action
        final_action = np.argmax(actions_q)
        return final_action
