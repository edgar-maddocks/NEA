from nea.agent import AlphaZero
from nea.network import AlphaModel
from nea.ml.nn import SGD

if __name__ == "__main__":
    a = AlphaZero(
        optimizer=SGD,
        mcts_epochs=3,
        n_example_games=500,
        nn_epochs=3,
        n_compare_games=5,
        eec=0.75,
        n_mcts_searches=50,
        replace_win_pct_threshold=60,
        save=True,
    )
    n = AlphaModel(n_res_layers=5, num_hidden_conv=32)
    a.train(n)
