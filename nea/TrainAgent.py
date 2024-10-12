from nea.agent import AlphaZero
from nea.network import AlphaModel
from nea.ml.nn import SGD

if __name__ == "__main__":
    a = AlphaZero(
        optimizer=SGD,
        mcts_epochs=10,
        n_example_games=100,
        nn_epochs=10,
        batch_size=32,
        n_compare_games=5,
        eec=1.41,
        n_mcts_searches=50,
        replace_win_pct_threshold=50,
        save=True,
    )
    n = AlphaModel(n_res_layers=5, num_hidden_conv=32)
    a.train(n)
