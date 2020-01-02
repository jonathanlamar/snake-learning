from itertools import product
from numpy.random import normal, randint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from time import sleep
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

# My stuff
from config.init_config import InitConfig

class Player(InitConfig):
    """
    This class mainly holds a keras model with methods for reading a game
    state and deciding how to move, as well as methods for breeding with
    another Player instance.
    """
    def __init__(self, weights=None):

        # Grab global config variables
        super().__init__()

        # Architecture parameters set in InitConfig
        layers = [ Dense(24, input_shape=(24,)) ]
        for _ in range(self.num_hidden_layers):
            layers.append(Dense(self.hidden_layer_size))
        layers.append(Dense(4))

        self.model = Sequential(layers)

        # I have to specify a loss in order to compile, even though I won't
        # be performing any kind of gradient descent.
        self.model.compile(loss='mse')

        if weights is not None:
            self.model.set_weights(weights)


    ###########################################################################
    #           Methods for interacting with a GameState instance
    ###########################################################################
    def parse_game_state(self, game_state):
        # Look in 8 directions (N, NE, E, SE, S, SW, W, NW) for wall, prize,
        # and body.  Return one-hot encoded presence of prize and body, and
        # inverse of distance to wall in all 8 directions (as an array of
        # length 24).

        # Hard coding direction order
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        inputs = []

        for dy, dx in directions:

            # Line of sight: an array of game booaord values starting from the
            # head, and going in this direction all the way to the nearest wall.
            LOS = game_state.get_line_of_sight(dy, dx)

            # Distance to wall represented by 1+len(LOS)
            dist_to_wall = len(LOS) + 1
            # Trying inverse distance now
            inputs.append( 1.0 / dist_to_wall )
            # Distance to prize
            inputs.append(self._one_hot_detect(LOS == -1))
            # Distance to body
            inputs.append(self._one_hot_detect(LOS > 0))

        return np.array(inputs).reshape(1, 24)


    def _one_hot_detect(self, line_of_sight):
        # Expects: A numpy array of booleans
        # Returns: 1 if any value is true, 0 otherwise
        return int(any(line_of_sight))


    def _dist_detect(self, line_of_sight):
        # Expects: A numpy array of booleans
        # Returns: index of first True from 0.  Length of array if none exist.

        target_locs, = np.where(line_of_sight)
        if len(target_locs) == 0:
            # If not found, then report index of the "horizon", which is
            # always board_size away, no matter where the head is located.
            target_index = self.board_size - 1
        else:
            target_index = target_locs[0]

        # Add 1 because LOS excludes snake head
        target_distance = target_index + 1
        return target_distance


    def decide_direction(self, parsed_game_state):
        # Expects: np array of shape (1, 24)
        # representing the 8 triples of parsed lines of sight generated by
        # parse_game_state.
        # Returns: 2D array representing dy, dx for input into GameState.update
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
            return np.array([0, 1])
        else:
            raise RuntimeError('Some weird argmax.')


    def play_game(self, game_state, draw_game=False, limit_time=True):
        # Play game until dead
        # Expects: GameState instance.
        # Returns: GameState

        time_limit = self.max_time_no_score if limit_time else np.inf
        previous_score = 0

        while (not game_state.dead) and (game_state.duration < time_limit):
            model_input = self.parse_game_state(game_state)
            new_direction = self.decide_direction(model_input)

            game_state.update(new_direction)

            if draw_game:
                game_state.draw()

            if limit_time and (game_state.score > previous_score):
                time_limit = min(self.max_time_allowed,
                                 time_limit + self.extra_time_per_score)
                previous_score = game_state.score
            elif game_state.score > previous_score:
                # Don't update time limit in this case.
                previous_score = game_state.score

        return game_state


    ###########################################################################
    #               Methods for making new Player instances
    ###########################################################################
    def breed(self, other, mutation_rate=None):
        # Combine with other and return new SnakeModel
        # Weights come as a list of arrays, one for each layer
        these_weights = self.model.get_weights()
        those_weights = other.model.get_weights()

        new_weights = []
        for left_array, right_array in zip(these_weights, those_weights):
            child = self._cross_arrays(left_array, right_array)
            child_mutated = self._mutate_array(child, mutation_rate)
            new_weights.append(child_mutated)

        return Player(weights=new_weights)


    def _cross_arrays(self, tensor1, tensor2):
        # Expects: Two numpy arrays of same shape.
        # Returns: "crossed" array, which is of the same shape
        # Select random index.  Everything before in lexicographic order comes
        # from tensor1, everything after comes from tensor2

        if tensor1.shape != tensor2.shape:
            raise RuntimeError('Incompatible matrix shapes for crossover.')

        num_entries = len(tensor1.flatten())
        split_point = randint(num_entries)

        crossed_tensor = np.zeros(num_entries)
        crossed_tensor[:split_point] = tensor1.flatten()[:split_point]
        crossed_tensor[split_point:] = tensor2.flatten()[split_point:]

        return crossed_tensor.reshape(tensor1.shape)


    def _mutate_array(self, arr, mutation_rate=None):
        # Mutate weights by adding gaussian noise
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        return arr + normal(scale = mutation_rate, size=arr.shape)


    def save_weights(self, save_loc):
        self.model.save_weights(save_loc)


    def load_weights(self, load_loc):
        self.model.load_weights(load_loc)
