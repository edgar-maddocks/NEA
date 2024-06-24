from ConsoleCheckers.CheckersGame import CheckersGame

game = CheckersGame()

done = False
while not done:
    game.render()
    valids = game.get_all_valid_moves()
    print("VALID MOVES: ")
    valids = valids["takes"] if len(valids["takes"]) > 0 else valids["simple"]
    for action in valids:
        print(CheckersGame.convert_rowcol_to_user(*action[0]), CheckersGame.convert_rowcol_to_user(*action[1]))

    user_action = input("Enter move in form numLETTERnumLETTER: ")
    action = ((int(user_action[0]), user_action[1]), (int(user_action[2]), user_action[3]))
    action = ((CheckersGame.convert_rowcol_to_game(*action[0])), (CheckersGame.convert_rowcol_to_game(*action[1])))
    valid, next_obs, done, reward = game.step(action)
    if reward == 1:
        print(f"{game.player} WON")
    elif reward == -1:
        print(f"{game._opposite_player} WON")
    elif done and reward == 0:
        print("DRAW")


    

    


    