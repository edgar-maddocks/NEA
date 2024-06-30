from ConsoleCheckers.CheckersGame import CheckersGame
from MCTS.MCTS import MCTS

done = False
game = CheckersGame()
user_colour = input("Please enter what colour you would like to play (w/b): ")
user_colour = "white" if user_colour == "w" else "black"
num_searches = int(input("Enter the num searches for the MCTS: "))
mcts = MCTS(eec=1.41, n_searches=num_searches, n_jobs=3)
last_action = None
while not done:
    game.render()
    if last_action is not None:
        last_action = CheckersGame.convert_rowcol_to_user(
            *last_action[0]
        ), CheckersGame.convert_rowcol_to_user(*last_action[1])
        print("LAST ACTION: ", last_action)
    if game.player == user_colour:
        valids = game.get_all_valid_moves()
        print("VALID MOVES: ")
        valids = valids["takes"] if len(valids["takes"]) > 0 else valids["simple"]
        for action in valids:
            print(
                CheckersGame.convert_rowcol_to_user(*action[0]),
                CheckersGame.convert_rowcol_to_user(*action[1]),
            )

        user_action = input("Enter move in form numLETTERnumLETTER: ")
        action = (
            (int(user_action[0]), user_action[1]),
            (int(user_action[2]), user_action[3]),
        )
        action = (
            (CheckersGame.convert_rowcol_to_game(*action[0])),
            (CheckersGame.convert_rowcol_to_game(*action[1])),
        )
        valid, next_obs, done, reward = game.step(action)
        last_action = action
    else:
        print("Building Tree...")
        mcts.mp_build_tree(game)
        action = mcts.get_action()
        print(
            f"WHITE'S MOVE:\n FROM:{CheckersGame.convert_rowcol_to_user(*action[0])}\n TO:{CheckersGame.convert_rowcol_to_user(*action[1])}"
        )
        valid, next_obs, done, reward = game.step(action)
        last_action = action

if reward == 1:
    print(f"{game.player} WON")
elif reward == -1:
    print(f"{game._opposite_player} WON")
elif done and reward == 0:
    print("DRAW")
