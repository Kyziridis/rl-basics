'''
write a simple Q-learning player
'''

class QAgent():
    def __init__(self, game, episodes, iter):
        self.Q = {}
        self.episodes = episodes
        self.game = game
        self.e = 0.2
        self.dc = 0.5
        self.e_ = 0.01


    def update(self):
        # Bellman

    def e_greedy(self, board, temp):
        s = str(board.flatten())

        if np.random.rand() <= self.e :
            possible_acts = np.arange(self.game.getActionSize())
            valids = self.game.getValidMoves(init_board, 1)
            valid_acts = possible_acts[valids==1]
            action = np.random.choice(valid_acts)
            return action

        actions_q = []
        for state in temp:
            actions_q.append(self.Q[state])

        action = np.argmax(actions_q)
        return action

    def opponent_play(self, board):
        possible_acts = np.arange(self.game.getActionSize())
        valids = self.game.getValidMoves(init_board, curPlayer)
        valid_acts = possible_acts[valids==1]
        action = np.random.choice(valid_acts)
        board, curPlayer = self.game.getNextState(board, curPlayer, action)
        return(board, curPlayer)

    def simulate(self):
        temp = []
        for ep in self.episodes:
            curPlayer = 1
            if ep == 0:
                self.Q[s] = 0.

            init_board = self.game.getInitBoard()
            s = str(init_board.flatten())

            possible_acts = np.arange(self.game.getActionSize())
            valids = self.game.getValidMoves(init_board, 1)
            valid_acts = possible_acts[valids==1]

            board = init_board
            while self.game.getGameEnded(board, 1) != 0:

                for action in valid_acts:
                    next_s, _ = self.game.getNextState(board, 1, action)
                    s = str(next_s.flatten())
                    temp.append(s)
                    if s not in self.Q:
                        self.Q[s] = 0.

                action = self.e_greedy(board, temp)
                temp = []
                board, curPlayer = self.game.getNextState(board, curPlayer, action)

                # check if my action resulted in terminal state
                if self.game.getGameEnded(board, 1) != 0:
                    ret = self.game.getGameEnded(board, 1)
                    Q_prime = ret
                    self.update(ret, Q_prime)
                    break

                board, curPlayer = self.opponent_play(board, curPlayer)

                # check if opponent's action resulted in terminal state
                if self.game.getGameEnded(board, 1) != 0:
                    ret = self.game.getGameEnded(board, 1)
                    Q_prime = ret
                    self.update(ret, Q_prime)
                    break

                # max Q(S', A')
                # check if new state is terminal

                possible_acts = np.arange(self.game.getActionSize())
                valids = self.game.getValidMoves(init_board, 1)
                valid_acts = possible_acts[valids==1]

                for action in valid_acts:
                    next_s, _ = self.game.getNextState(board, 1, action)
                    s = str(next_s.flatten())
                    temp.append(s)
                    if s not in self.Q:
                        self.Q[s] = 0.

                qs = []
                for state in temp:
                    qs.append(self.Q[state])
                Q_prime = max(qs)

                ret = 0
                self.update(ret, Q_prime)




    def play(self, board):
        #
