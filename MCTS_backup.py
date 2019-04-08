'''
write MCTS player by yourself
'''
import numpy as np

	
# gamw ti panagia 

class MCTSAgent():
    def __init__(self, game, iters, c):
        self.game = game
        self.S = {}
        self.gameEnded = {}
        self.iters = iters
        self.c = c
        self.state_stats = {} 

    def UCB(self, q, N, n, epsilon=1e-8):
        return q + self.c* np.sqrt(np.log(N + epsilon) / (n + epsilon))

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
        return self.game.getGameEnded(board, 1)    

    def traverse(self, valids, board):
        print('Traverse valids', valids)
        possible_acts = np.arange(self.game.getActionSize()) # logw autounou
        valid_acts = possible_acts[valids==1]
        neg_acts = possible_acts[valids==0]
        ucb_values = np.zeros(self.game.getActionSize())
        ucb_values[neg_acts] = -1e+11 
        for a in valid_acts:
            next_s, _ = self.game.getNextState(board, 1, a)
            next_s = str(next_s.flatten())
            
            if next_s not in self.state_stats:
                print('Mpika next_s')
                self.state_stats[next_s] = np.array([0,0])
                
            ucb_values[a] = self.UCB(self.state_stats[next_s][1], self.iters, self.state_stats[next_s][0])
            # ucb_values.append(self.UCB(self.state_stats[next_s][1], self.iters, self.state_stats[next_s][0]))
        print('\nucb', ucb_values)    
        action = np.argmax(ucb_values)
        return action

    def simulate(self, board):
        # Get current state and check if it is Terminal_State
        s = str(board.flatten())
        print(s)
        if self.game.getGameEnded(board, 1) != 0:
            q = self.game.getGameEnded(board, 1)
            return q
        
        # Check if the current state is visited and call rollout
        if s not in self.S:
            # koita to count tou s
            # diaforetiko dict gia to count mi gamisw kamia panagia!
            self.S[s] = 1
            if self.sim == 0:
                print('Prwto iter MONO')
                self.state_stats[s] = np.array([0,0])

            #self.state_stats[s] = np.array([0,0])
            q = self.rollout(board)
            print('Rollout: ', q)    
            return q
        
        # Take valid actions and traverse
        valids = self.game.getValidMoves(board, 1)
        action = self.traverse(valids, board)
        print('\naction', action)
        # Play the action in the MCTS_simulation and go to the next state
        next_s, curPlayer = self.game.getNextState(board, 1, action)
        next_s = self.game.getCanonicalForm(next_s, curPlayer)
        
        # Recursion
        q = self.simulate(next_s)

        # UPDATE put a function in the future
        print("\nUPDate")
        self.state_stats[s][0] += 1
        self.state_stats[s][1] = (self.state_stats[s][1] + q) / self.state_stats[s][0] 
        print('stats:' , self.state_stats)
        return q

    def play(self, board):
        for self.sim in range(self.iters):
            print("\niter: " + str(self.sim))
            self.simulate(board)
        
                