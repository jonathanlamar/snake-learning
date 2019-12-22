# My stuff
from ai.init_config import InitConfig
from ai.player import Player

class Generation(InitConfig):
    """
    This class is responsible for managing the spawning and ranking of one
    generation of player instances, as well as spawning its successor
    generation.
    """
    def __init__(self, previous_gen=None):

        # Grab global config variables
        super().__init__()

        # This will be set by playing the game
        self.scores = None

        # Either breed previous gen or start fresh
        if previous_gen is not None:
            breeders = previous_gen.get_breeders()
            self.players = self.spawn(breeders)
        else:
            self.players = [Player() for i in range(self.generation_size)]


        def get_breeders(self):
            pass

        def spawn(self, breeders):
            pass

        def eval_players(self):
            pass

        def score_one_player(self):
            pass
