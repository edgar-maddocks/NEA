from TicTacToe import TicTacToe

class Node:
    def __init__(self, game, state, args, parents=None, action_taken = None):
        self.game = game
        self.state = state
        self.args = args
        self
        

class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        
        for search in range(self.args["num_searches"]):

