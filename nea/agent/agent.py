import numpy as np
from tqdm import tqdm

from collections import deque
import gc
import itertools
import random

from nea.ml.nn import Optimizer, SGD, AlphaLoss
from nea.ml.autograd import Tensor
from nea.mcts import AlphaMCTS, AlphaNode
from nea.network import AlphaModel
from nea.console_checkers import CheckersGame
from nea.agent.memory_types import SAP, SPV
from nea.agent.consts import ACTION_SPACE


class AlphaZero:
    def __init__(
        self,
        optimizer: Optimizer = None,
        mcts_epochs: int = 50,
        n_example_games: int = 10,
        max_training_examples: int = 500,
        nn_epochs: int = 10,
        batch_size: int = 32,
        n_compare_games: int = 10,
        eec: float = 1.41,
        n_mcts_searches: int = 100,
        replace_win_pct_threshold: int = 59,
    ) -> None:
        self.prev_model = None
        self.new_model = None
        if optimizer is None:
            optimizer = SGD
        self.optimizer = optimizer
        self.hyperparams = {
            "mcts_epochs": mcts_epochs,
            "n_example_games": n_example_games,
            "max_training_examples": max_training_examples,
            "nn_epochs": nn_epochs,
            "batch_size": batch_size,
            "n_compare_games": n_compare_games,
            "eec": eec,
            "n_mcts_searches": n_mcts_searches,
            "replace_win_pct_threshold": replace_win_pct_threshold,
        }

        self.loss = AlphaLoss()

    def train(self, initialModel: AlphaModel) -> None:
        self.prev_model = initialModel
        if self.new_model is None:
            self.new_model = initialModel

        for mcts_epoch in range(self.hyperparams["mcts_epochs"]):
            training_examples = deque(maxlen=self.hyperparams["max_training_examples"])

            gc.collect()
            print("GETTING EXAMPLE GAMES")
            for example in tqdm(range(int(self.hyperparams["n_example_games"]))):
                gc.collect()

                game_saps, reward, player = self._get_example_saps()
                game_spvs = self._convert_saps_to_spvs(game_saps, player, reward)

                for item in game_spvs:
                    training_examples.append(item)

                    if (
                        len(training_examples)
                        >= self.hyperparams["max_training_examples"]
                    ):
                        break

            print("BEGINNING NN TRAINING")
            for epoch in tqdm(range(int(self.hyperparams["nn_epochs"]))):
                gc.collect()
                self._train_nn(training_examples)

            print("PLAYING COMPARISON GAMES")
            updated_model = False

            if self._play_compare_games():
                updated_model = True
                self.prev_model = self.new_model

            gc.collect()

        return self.prev_model, updated_model

    def _get_example_saps(self) -> tuple[deque[SAP], float, str]:
        game = CheckersGame()

        mcts = AlphaMCTS(
            self.prev_model,
            eec=self.hyperparams["eec"],
            n_searches=self.hyperparams["n_mcts_searches"],
        )

        game_saps = deque()
        prior_states = deque(maxlen=5)

        done = False
        while not done:
            prior_states.append(game.board)

            mcts.alpha_build_tree(game, prior_states)
            action_probs = mcts.get_action_probs()

            if len(prior_states) < 4:
                idx_choice = np.random.choice(8)
                x_choice = np.random.choice(8)
                y_choice = np.random.choice(8)

                random_action = ACTION_SPACE[idx_choice, x_choice, y_choice]

                action = AlphaNode._convert_action_idx_to_action_game(random_action)
            else:
                action = mcts.convert_probs_to_action(action_probs)

            game_saps.append(SAP(game.board, action_probs, game.player))

            _, _, done, reward = game.step(action=action)

        return game_saps, reward, game.player

    def _convert_saps_to_spvs(
        self, game_saps: deque[SAP], player: str, reward: float
    ) -> deque[SPV]:
        len_game_saps = len(game_saps)
        game_spvs = deque(maxlen=len_game_saps)

        for item in game_saps:
            value = reward if item.player == player else reward * -1
            game_spvs.append(SPV(item.state, item.mcts_action_probs, value))

        return game_spvs

    def _train_nn(self, example_games: deque[SPV]) -> deque[SPV]:
        random.shuffle(example_games)
        len_example_games = len(example_games)
        optimizer = self.optimizer(self.new_model.params)

        for batch_idx in range(0, len_example_games, self.hyperparams["batch_size"]):
            end_idx = min(
                len_example_games - 1, batch_idx + self.hyperparams["batch_size"]
            )
            batch = list(itertools.islice(example_games, batch_idx, end_idx))

            past_states = []
            loss = 0
            for i in range(0, len(batch)):
                spv = batch[i]
                past_states.append(spv.state)

                if len(past_states) > 5:
                    past_states.pop(0)

                if i >= 5:
                    state, mcts_probs, true_value = (
                        Tensor(past_states, requires_grad=True),
                        Tensor(spv.mcts_action_probs),
                        Tensor(spv.true_value),
                    )

                    nn_policy, nn_value = self.new_model(state)

                    loss += self.loss(true_value, nn_value, mcts_probs, nn_policy)

            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _play_compare_games(self) -> bool:
        new_nn_wins = 0
        prev_nn_wins = 0

        for compare_game in tqdm(range(int(self.hyperparams["n_compare_games"]))):
            gc.collect()
            game = CheckersGame()

            prev_mcts = AlphaMCTS(
                self.prev_model,
                eec=self.hyperparams["eec"],
                n_searches=self.hyperparams["n_mcts_searches"],
            )
            new_mcts = AlphaMCTS(
                self.new_model,
                eec=self.hyperparams["eec"],
                n_searches=self.hyperparams["n_mcts_searches"],
            )

            prior_states = deque(maxlen=5)

            done = False
            prev_nn_player = np.random.choice(["white", "black"], 1)

            while not done:
                prior_states.append(game.board)

                if game.player == prev_nn_player:
                    prev_mcts.alpha_build_tree(game, prior_states)
                    action = prev_mcts.get_action()

                    valid, _, done, reward = game.step(action)
                    if not valid:
                        print("TRIED TO MAKE INVALID MOVE")
                    if done and reward == 1:
                        prev_nn_wins += 1
                    elif done and reward == -1:
                        new_nn_wins += 1

                else:
                    new_mcts.alpha_build_tree(game, prior_states)
                    action = new_mcts.get_action()

                    valid, _, done, reward = game.step(action)
                    if not valid:
                        print("TRIED TO MAKE INVALID MOVE")
                    if done and reward == 1:
                        new_nn_wins += 1
                    elif done and reward == -1:
                        prev_nn_wins += 1

        new_nn_win_pct = (new_nn_wins / prev_nn_wins) * 100
        if new_nn_win_pct > self.hyperparams["replace_win_pct_threshold"]:
            return True
        return False


if __name__ == "__main__":
    alphazero = AlphaZero(SGD, mcts_epochs=10)
    model = AlphaModel()
    model, updated_model = alphazero.train(model)
    if updated_model:
        print("MODEL UPDATED")
