'''
write MCTS player by yourself
'''
import numpy as np

	


class MCTSAgent():
    def __init__(self, game, iters, c):
        self.game = game
        self.S = {}
        self.gameEnded = {}
        self.iters = iters
        self.c = c
    def UCB(self, q, N, n, epsilon=1e-8):
        return q + self.c* np.sqrt(np.log(N)/n + epsilon)

    def rollout(self, board):
        curPlayer = 1
        valids = self.game.getValidMoves(board, 1)
        valid_acts = np.where(valids == 1)[0]
        selected_act = np.random.choice(valid_acts, replace=False)
        board, curPlayer = self.game.getNextState(board, curPlayer, selected_act)
        while self.game.getGameEnded(board, curPlayer)==0:
            valids = self.game.getValidMoves(board, 1)
            valid_acts = np.where(valids == 1)[0]
            selected_act = np.random.choice(valid_acts, replace=False)
            board, curPlayer = self.game.getNextState(board, curPlayer, selected_act)
        return(self.game.getGameEnded(board, 1))    


    def traverse(self, valids):
        possible_acts = np.arange(self.game.getActionSize())[valids==1]
        ucb_values = []
        for a in possible_acts:
            ucb_values.append(self.UCB(self.S[s][1], self.iters, self.S[s][0]))
        action = np.argmax(ucb_values)
        return action
    def Update():        

    def simulate(self, board):
        # Get current state and check if it is Terminal_State
        s = str(board.flatten())
        if self.game.getGameEnded(board, 1) != 0:
            q = self.game.getGameEnded(board, 1)
            return q
        
        
        # Check if the current state is visited and call rollout
        if s not in self.S:
            self.S[s] = np.array([0, 0])
            q = self.rollout(board)
        
        # Take valid actions and traverse
        valids = self.game.getValidMoves(board, 1)
        action = self.traverse(valids)

        # Play the action in the MCTS_simulation and go to the next state
        next_s, curPlayer = self.game.getNextState(board, 1, action)
        next_s = self.game.getCanonicalForm(next_s, curPlayer)

        # UPDATE
        
    def play(self, board):
        for sim in range(self.iters):
            self.simulate(board)
        
        
        
        
        