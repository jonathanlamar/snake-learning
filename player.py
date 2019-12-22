import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from numpy.random import normal, randint
from init_config import InitConfig

class Player(InitConfig):
    """
    This class mainly holds a keras model (for now, fixed geometry),
    with methods for reading a game state and deciding how to move, as well as
    methods for breeding with another Player instance.
    """
    def __init__(self, weights=None):

        # Grab global config variables
        super().__init__()

        # TODO: Try different activation functions.
        self.model = Sequential([
            Dense(24, input_shape=(24,)),
            Dense(18),
            Dense(18),
            Dense(4),
        ])

        # I have to specify a loss, even though I won't be "training" the model.
        self.model.compile(loss='mse')

        if weights is not None:
            self.model.set_weights(weights)


    ###########################################################################
    #           Methods for interacting with a GameState instance
    ###########################################################################
    def parse_game_state(self, game_state):
        # TODO: In 8 different directions, we need to read distance to prize,
        # distance to wall, and distance to self.
        pass

    def decide_direction(self, parsed_game_state):
        out_arr = self.model.predict(parsed_game_state)
        direction = out_arr.argmax()

        if direction == 0:
            # Return an ordered pair representing dy, dx in array ordering
            # (i.e., -1,0 means Up)
            return np.array([-1, 0])
        elif direction == 1:
            # Down
            return np.array([1, 0])
        elif direction == 2:
            # Left
            return np.array([0, -1])
        elif direction == 3:
            # Right
            return np.array([1, 1])

    ###########################################################################
    #               Methods for making new Player instances
    ###########################################################################
    def breed(self, other):
        # Combine with other and return new SnakeModel
        # Weights come as a list of arrays, one for each layer
        these_weights = self.model.get_weights()
        those_weights = other.model.get_weights()

        new_weights = []
        for left_array, right_array in zip(these_weights, those_weights):
            child = self._cross_arrays(left_array, right_array)
            child_mutated = self._mutate_array(child)
            new_weights.append(child_mutated)

        return Player(weights=new_weights)

    def _cross_arrays(self, tensor1, tensor2):
        # Starting completely random for now.  The other person took a rectangle
        # from one side and the rest from the other.
        crossed_tensor = np.zeros(shape=tensor1.shape)

        nrows, ncols = tensor1.shape
        if tensor2.shape[0] != nrows or tensor2.shape[1] != ncols:
            raise RuntimeError('Incompatible matrix shapes for crossover.')

        randR = randint(0, nrows)
        randC = randint(0, ncols)

        for i in range(nrows):
            for j in range(ncols):
                if i < randR or (i == randR and j <= randC):
                    crossed_tensor[i, j] = tensor1[i, j]
                else:
                    crossed_tensor[i, j] = tensor2[i, j]

        return crossed_tensor

    def _mutate_array(self, arr):
        # Mutate weights by adding gaussian noise
        return arr + normal(scale = self.mutation_rate, size=arr.shape)

