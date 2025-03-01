from collections import deque

import numpy as np

from nea.mcts import AlphaMCTS
from nea.console_checkers import CheckersGame
from nea.network import AlphaModel


def simGames(
    n_games: int = 10,
    n_searches_mcts1: int = 100,
    n_searches_mcts2: int = 100,
    eec_mcts1: float = 1.41,
    eec_mcts2: float = 1.41,
    verbose: int = 0,
):
    """Simulates a given number of games between two mcts models with
    given hyperparams

    Args:
        n_games (int, optional): Defaults to 10.
        n_searches_mcts1 (int, optional): hyperparameter. Defaults to 100.
        n_searches_mcts2 (int, optional): hyperparameter. Defaults to 100.
        eec_mcts1 (float, optional): hyperparameter. Defaults to 1.41.
        eec_mcts2 (float, optional): hyperparameter. Defaults to 1.41.
        verbose (int, optional): 1 to display info. Defaults to 0.

    Returns:
        list[str]: list of which model won each game
    """
    games = []
    for gamen in range(n_games):
        prior_states = deque(maxlen=5)
        alphamcts1 = AlphaMCTS(
            model=AlphaModel(), eec=eec_mcts1, n_searches=n_searches_mcts1
        )
        alphamcts2 = AlphaMCTS(
            model=AlphaModel(), eec=eec_mcts2, n_searches=n_searches_mcts2
        )

        game = CheckersGame()

        done = False
        mcts1_player = np.random.choice(["white", "black"], 1)

        while not done:
            prior_states.append(game.board)
            if verbose:
                game.render()
            print("GAME: ", gamen + 1)

            if game.player == mcts1_player:
                if verbose:
                    print("Building Tree...")
                alphamcts1.alpha_build_tree(game, prior_states)
                action = alphamcts1.get_action()

                if verbose:
                    print(
                        f"""WHITE'S MOVE:\n FROM:{CheckersGame.convert_rowcol_to_user(*action[0])}
                        TO:{CheckersGame.convert_rowcol_to_user(*action[1])}"""
                    )

                valid, _, done, reward = game.step(action)
                if not valid and verbose:
                    print("TRIED TO MAKE INVALID MOVE")
                if done and reward == 1:
                    games.append("mcts1")
                elif done and reward == 0:
                    games.append("draw")
                elif done and reward == -1:
                    games.append("mcts2")

            else:
                if verbose:
                    print("Building Tree...")
                alphamcts2.alpha_build_tree(game, prior_states)
                action = alphamcts2.get_action()

                if verbose:
                    print(
                        f"""BLACK'S MOVE:\n FROM:{CheckersGame.convert_rowcol_to_user(*action[0])}
                        TO:{CheckersGame.convert_rowcol_to_user(*action[1])}"""
                    )

                valid, _, done, reward = game.step(action)
                if not valid and verbose:
                    print("TRIED TO MAKE INVALID MOVE")
                if done and reward == 1:
                    games.append("mcts2")
                elif done and reward == 0:
                    games.append("draw")
                elif done and reward == -1:
                    games.append("mcts1")

    print(games)
    print(f"mcts1 was {mcts1_player}")
    return games


if __name__ == "__main__":
    simGames(1, n_searches_mcts1=100, n_searches_mcts2=10, verbose=1)
