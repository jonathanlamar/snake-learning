import numpy as np


class InitConfig:
    """
    This class holds everything intended to be configurable by the user in one
    location.  I'm doing this instead of a run script with global variables.
    """

    def __init__(self) -> None:

        # Related to generations and breeding
        # Total population per generation
        self.generation_size = 2000
        # Number of top players to breed next generation
        self.number_to_breed = 10
        # Standard deviation of Gaussian noise added during breeding algorithm
        self.mutation_rate = 0.2
        # Take average of this many games to select best players
        self.num_games_to_play = 1

        # Related to halting the game:
        # This many frames with no score = kill
        self.max_time_no_score = 500
        # Each new score allows this many more frames
        self.extra_time_per_score = 100
        # Overall max frames
        self.max_time_allowed = 3000

        # How big should the game be?
        self.board_size = 30

        # The neural network will have an input layer of size 24, output layer
        # of size 4, and num_hidden_layers input layers of size hidden_layer_size
        self.num_hidden_layers = 2
        self.hidden_layer_size = 18

    def fitness_function(self, score: int, duration: int) -> float:
        # This is the fitness function for the selection algorithm.
        # Generation instances will use this function to decide fitness.

        return 2 * np.log(1 + duration) + score
