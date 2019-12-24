import numpy as np

# My stuff
from config.init_config import InitConfig
from ai.player import Player
from game.game_state import GameState

class Generation(InitConfig):
    """
    This class is responsible for managing the spawning and ranking of one
    generation of player instances, as well as spawning its successor
    generation.
    """
    def __init__(self, breeders=None, gen_number=0):

        # Grab global config variables
        super().__init__()

        # This will be set by playing the game
        self.scores = None

        # Either breed previous gen or start fresh
        if breeders is not None:
            self.players = self.breed(breeders)
        else:
            # TODO: This is slow.
            self.players = np.array([Player() for _ in range(self.generation_size)])

        self.gen_number = gen_number


    def breed(self, breeders):
        new_gen = []
        for _ in range(self.generation_size):
            P1, P2 = np.random.choice(breeders, 2, replace=False)
            new_gen.append(P1.breed(P2))

        return np.array(new_gen)

    def get_breeders(self):
        top_k_inds = np.argsort(self.scores)[::-1][:self.number_to_breed]
        return self.players[top_k_inds]

    def eval_players(self):
        scores = []
        # This should be run in parallel
        for P in self.players:

            # Ugh, state.  This alters G in place.
            G = GameState()
            P.play_game(G)

            performance = G.score * self.score_weight + G.time * self.duration_weight
            scores.append(performance)

        self.scores = np.array(scores)

    def advance_next_gen(self):
        # Returns: New Generation instance bred from current.
        breeders = self.get_breeders()
        return Generation(breeders)

    def save_players(self, save_dir='data'):
        for i, P in enumerate(self.players):
            P.save_model_weights(save_dir + '/gen' + self.gen_number + '/_player' + i + '.h5')

    def load_players(self, save_dir='data', use_gen=True):
        for i, P in enumerate(self.players):
            if use_gen:
                P.load_model_weights(save_dir + '/gen' + self.gen_number + '/_player' + i + '.h5')
            else:
                P.load_model_weights(save_dir + '/_player' + i + '.h5')
