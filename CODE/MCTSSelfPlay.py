from MCTS.MCTS import MCTS
from ConsoleCheckers.CheckersGame import CheckersGame

import numpy as np

def simGames(n_games: int = 10, 
            n_searches_mcts1: int = 100000, 
            n_searches_mcts2: int = 10000, 
            eec_mcts1: float = 1.41, 
            eec_mcts2: float = 1.41,
            verbose: int = 0):
            games = []
            for gamen in range(n_games):
                mcts1 = MCTS(eec=eec_mcts1, n_searches= n_searches_mcts1)
                mcts2 = MCTS(eec=eec_mcts2, n_searches=n_searches_mcts2)

                game = CheckersGame()

                done = False
                mcts1_player = np.random.choice(["white", "black"], 1)

                while not done:
                    if verbose:
                        game.render()
                    print("GAME: ", gamen)

                    if game.player == mcts1_player:
                        if verbose:
                            print("Building Tree...")
                        mcts1.build_tree(game)
                        action = mcts1.get_action()
                        if verbose:
                            print(
                                f"WHITE'S MOVE:\n FROM:{CheckersGame.convert_rowcol_to_user(*action[0])}\n TO:{CheckersGame.convert_rowcol_to_user(*action[1])}"
                            )
                        valid, next_obs, done, reward = game.step(action)
                        if not valid and verbose:
                            print("TRIED TO MAKE INVALID MOVE")
                        if done and reward == 1:
                            games.append("mcts1")
                        elif done and reward == 0:
                            games.append("draw")
                        elif done and reward == -1:
                            games.append("mcts2")
                    
                    else:
                        mcts2.build_tree(game)
                        action = mcts2.get_action()
                        if verbose:
                            print(
                                f"BLACK'S MOVE:\n FROM:{CheckersGame.convert_rowcol_to_user(*action[0])}\n TO:{CheckersGame.convert_rowcol_to_user(*action[1])}"
                            )
                        valid, next_obs, done, reward = game.step(action)
                        if not valid and verbose:
                            print("TRIED TO MAKE INVALID MOVE")
                        if done and reward == 1:
                            games.append("mcts2")
                        elif done and reward == 0:
                            games.append("draw")
                        elif done and reward == -1:
                            games.append("mcts1")

            print(games)
            return games

simGames(10)

