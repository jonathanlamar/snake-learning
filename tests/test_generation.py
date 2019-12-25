import numpy as np
import pandas as pd

# My stuff
from ai.generation import Generation

def test_load_generation():
    def player_equality(P, Q):
        return all([np.all(x == y) for x, y in zip(P.model.get_weights(),
                                                   Q.model.get_weights())])

    G = Generation(generation_size=10)
    G.spawn_random()
    G.eval_players()
    G.save_latest_gen(test=True)

    H = Generation(generation_size=10)
    H.load_latest_gen(test=True)

    assert all([player_equality(P, Q)
                for P, Q in zip(G.players, H.players)])

def test_repeat_performance():
    G = Generation(generation_size=10)
    G.spawn_random()
    G.eval_players()

    df = G.summary
    G.summary = pd.DataFrame()

    # Make sure df is nonempty
    assert df.shape[0] != 0 and df.shape[1] != 0

    G.eval_players()
    assert np.all(df == G.summary)
