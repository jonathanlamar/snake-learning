import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# This class mainly holds a keras model (for now, fixed geometry),
# with methods for initializing, mutating, and breeding
class SnakeModel:
    def __init__(self, weights=None, mutate_factor=None):
        # TODO: Try different activation functions.
        self.model = Sequential([
            Dense(24, input_shape=(24,)),
            Dense(16),
            Dense(16),
            Dense(4),
        ])

        # I have to specify a loss, even though I won't be "training" the model.
        self.model.compile(loss='mse')

        if weights is not None:
            self.model.set_weights(weights)

        if mutate_factor is not None:
            self._mutate_weights(mutate_factor)

    def _mutate_weights(self, mutate_factor=0.1):
        # Mutate weights by adding gaussian noise
        weights = self.model.get_weights()

        for w in weights:
            w += np.random.normal(scale = mutate_factor, size=w.shape)

        self.model.set_weights(weights)

    def breed(self, other, mutate_factor):
        # Combine with other and return new SnakeModel
        these_weights = self.model.get_weights()
        those_weights = other.model.get_weights()

        new_weights = self._cross_tensors(these_weights, those_weights)

        return SnakeModel(weights=new_weights, mutate_factor=mutate_factor)

    def _cross_tensors(self, tensor1, tensor2):
        # Starting completely random for now.  The other person took a rectangle
        # from one side and the rest from the other.
        choose_matrix = np.random.binomial(1, 0.5, size=tensor1.shape)
        crossed_tensor = np.zeros(shape=tensor1.shape)

        # TODO: Test this.
        crossed_tensor[choose_matrix == 0] = tensor1[choose_matrix == 0]
        crossed_tensor[choose_matrix == 1] = tensor2[choose_matrix == 1]

        return crossed_tensor

    def play_and_score(self):
        # Play the snake game and return score and time for ranking
        # TODO
        pass
