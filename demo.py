from ai.player import Player
from game.game_state import GameState
from ai.generation import Generation
import numpy as np

if __name__ == "__main__":
    gen = Generation()
    gen.load_latest_gen()
    leaderboard = gen.get_leader_board()
    max_dur = leaderboard["duration"].max()
    player_num = leaderboard.loc[leaderboard["duration"] == max_dur, "model"].values[0]
    player = Player()
    player.load_weights(f"./data/gen{gen.gen_number:04.0f}/player{player_num:04.0f}.h5")

    seed = np.random.randint(1000)
    game = GameState(seed=seed)
    player.play_game(game, draw_game=True, limit_time=False)
    print("Seed = %d" % seed)
