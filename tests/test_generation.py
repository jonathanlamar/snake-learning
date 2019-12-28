import numpy as np
import pandas as pd

# My stuff
from ai.generation import Generation
from game.game_state import GameState

def test_repeatable():
    def player_equality(P, Q):
        P_weights = P.model.get_weights()
        Q_weights = Q.model.get_weights()
        L = [np.all(p == q) for p, q in zip(P_weights, Q_weights)]
        return all(L)

    gen = Generation(generation_size=10)
    gen.spawn_random()
    gen.eval_players()
    gen.save_latest_gen(test=True)

    gen2 = Generation(generation_size=10)
    gen2.load_latest_gen(test=True)

    # Does H have the same players
    assert all([player_equality(P, Q) for P, Q in zip(gen.players, gen2.players)])

    df = gen2.summary
    scores = []
    durations = []
    for i in range(gen2.generation_size):
        P = gen2.players[i]
        seed = df.loc[df['model']==i, 'seed'].values[0]
        G = GameState(seed=seed)
        P.play_game(G)

        scores.append(G.score)
        durations.append(G.duration)

    assert np.all(np.array(scores) == df['score'].values)
    assert np.all(np.array(durations) == df['duration'].values)
