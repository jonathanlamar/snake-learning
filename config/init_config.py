class InitConfig:
    """
    This class holds only values and gets inherited by anything that needs it.
    I'm doing this instead of a run script with global variables.
    """
    def __init__(self):

        # Related to generations and breeding
        self.generation_size = 20
        self.number_to_breed = 5
        # Variance of Gaussian noise added during breeding algorithm
        self.mutation_rate = 1

        # Related to halting the game:
        # This many frames with no score = kill
        self.max_time_no_score = 100
        # Each new score allows this many more frames
        self.extra_time_per_score = 50
        # Overall max frames
        self.max_time_allowed = 500

        # These are the coefficients of score and game duration in the fitness
        # function, respectively.
        self.score_weight = 100
        self.duration_weight = 1

        # How big should the game be?
        self.board_size = 50

        # Seed for the random number generator.
        # I don't think this has any effect on the random weights for the
        # players, but does affect the placement of the apples.
        self.seed = 1234
