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

    def simulate(self):
        temp = []
        for ep in self.episodes:
            init_board = self.game.getInitBoard()
            s = str(init_board.flatten())

            possible_acts = np.arange(self.game.getActionSize())
            valids = self.game.getValidMoves(init_board, 1)
            valid_acts = possible_acts[valids==1]

            if ep == 0:
                self.Q[s] = 0.
            board = init_board
            while self.game.getGameEnded(board, 1) != 0:

                for action in valid_acts:
                    next_s,_ = self.game.getNextState(board, 1, action)
                    s = str(next_s.flatten())
                    if s not in self.Q:
                        temp.append(s)
                        self.Q[s] = 0.








    def play(self, board):
        #


