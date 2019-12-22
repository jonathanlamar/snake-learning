class InitConfig:
    """
    This class holds only values and gets inherited by anything that needs it.
    I'm doing this instead of a run script with global variables.
    """
    def __init__(self):
        # Variance of Gaussian noise added during breeding algorithm
        print("Initializing config")
        self.mutation_rate = 1

        # These are the coefficients of score and game duration in the fitness
        # function, respectively.
        self.score_weight = 100
        self.duration_weight = 1
