#!/usr/bin/env python3

from ai.player import Player
from game.game_state import GameState
import numpy as np

if __name__ == "__main__":
    player = Player()
    player.load_weights('./data/player0431.h5')

    seed = np.random.randint(1000)
    game = GameState(seed=seed)
    player.play_game(game, draw_game=True, limit_time=False)
    print("Seed = %d" % seed)
