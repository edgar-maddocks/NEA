import pickle
from collections import deque
import numpy as np

from nea.checkers_gui import CheckersGUI
from nea.checkers_gui.consts import DISPLAY, COLOURS
from nea.mcts import AlphaMCTS

import pygame


def agent_vs_agent_game_loop(
    n_searches_1: int = 50,
    eec_1: float = 0.75,
    example_games_1: int = 100,
    comparison_games_1: int = 5,
    replacement_threshold_1: int = 60,
    n_searches_2: int = 50,
    eec_2: float = 1.41,
    example_games_2: int = 500,
    comparison_games_2: int = 5,
    replacement_threshold_2: int = 60,
    agent_1_colour: str = "black",
    num_games: int = 1,
) -> list[str]:
    random_float = np.random.random()
    if random_float > 0.5:
        agent_1_colour = "white"
    elif random_float >= 0.5:
        agent_1_colour = "black"

    print(f"Agent 1 is {agent_1_colour}")

    pygame.init()

    screen = pygame.display.set_mode((DISPLAY.SCREEN_SIZE, DISPLAY.SCREEN_SIZE))
    wins = []

    for game_n in range(num_games):
        done = False

        gui = CheckersGUI()

        agent_1_file_path = (
            f"{n_searches_1}ns-"
            + f"{eec_1}ec-"
            + f"{example_games_1}te-"
            + f"{comparison_games_1}cg-"
            + f"{replacement_threshold_1}rt"
        )
        net_1 = None
        with open(f"nea/{agent_1_file_path}", "rb") as fh:
            net_1 = pickle.load(fh)

        agent_1 = AlphaMCTS(net_1, eec=eec_1, n_searches=n_searches_1)

        agent_2_file_path = (
            f"{n_searches_2}ns-"
            + f"{eec_2}ec-"
            + f"{example_games_2}te-"
            + f"{comparison_games_2}cg-"
            + f"{replacement_threshold_2}rt"
        )

        net_2 = None
        with open(f"nea/{agent_2_file_path}", "rb") as fh:
            net_2 = pickle.load(fh)

        agent_2 = AlphaMCTS(net_2, eec=eec_2, n_searches=n_searches_2)

        prior_states = deque(maxlen=5)

        winner = None
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    done = True
                    reward = "None"

            prior_states.append(gui.board)

            if gui.player == agent_1_colour:
                agent_1.alpha_build_tree(gui, prior_states)
                action = agent_1.get_action()

                done, reward = gui.evaluate_action(action)
            elif gui.player != agent_1_colour:
                agent_2.alpha_build_tree(gui, prior_states)
                action = agent_2.get_action()

                done, reward = gui.evaluate_action(action)

            if done:
                if reward == -1:
                    winner = gui.opposite_player
                    break
                elif reward == 1:
                    winner = gui.player
                    break
                elif reward == "None":
                    winner = "no one"
                    break

            screen.fill(COLOURS.BLACK)
            gui.draw(screen)
            pygame.display.flip()

        if winner == agent_1_colour:
            wins.append("agent 1")
        elif winner == "None":
            print("DRAW")
        else:
            wins.append("agent 2")

    num_agent_1_wins = len([x for x in wins if x == "agent 1"])
    num_agent_2_wins = len([x for x in wins if x == "agent 2"])

    if num_agent_2_wins == num_agent_1_wins:
        print(f"DRAW OVER {num_games} GAMES")
    elif num_agent_2_wins > num_agent_1_wins:
        print("AGENT 2 WINS")
        print("AGENT PARAMS: ")
        print(
            f"{n_searches_2}ns-"
            + f"{eec_2}ec-"
            + f"{example_games_2}te-"
            + f"{comparison_games_2}cg-"
            + f"{replacement_threshold_2}rt"
        )
    elif num_agent_1_wins > num_agent_2_wins:
        print("AGENT 1 WINS")
        print("AGENT PARAMS: ")
        print(
            f"{n_searches_1}ns-"
            + f"{eec_1}ec-"
            + f"{example_games_1}te-"
            + f"{comparison_games_1}cg-"
            + f"{replacement_threshold_1}rt"
        )


if __name__ == "__main__":
    agent_vs_agent_game_loop()
