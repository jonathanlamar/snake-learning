import numpy as np

class InitConfig:
    """
    This class holds only values and gets inherited by anything that needs it.
    I'm doing this instead of a run script with global variables.
    """
    def __init__(self):

        # Related to generations and breeding
        self.generation_size = 2000
        self.number_to_breed = 5
        # Standard deviation of Gaussian noise added during breeding algorithm
        self.mutation_rate = 0.8

        # Related to halting the game:
        # This many frames with no score = kill
        self.max_time_no_score = 500
        # Each new score allows this many more frames
        self.extra_time_per_score = 100
        # Overall max frames
        self.max_time_allowed = 1000

        # How big should the game be?
        self.board_size = 30

    def performance_metric(self, score, duration):
        # This is the performance metric for the selection algorithm.
        # Generation instances will use this function to decide fitness.

        return np.log(1 + 2*duration) + 2*score
