import numpy as np
import pandas as pd
import os

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
    def __init__(self, breeders=None, gen_number=1):

        # Grab global config variables
        super().__init__()

        # Dataframe with summary statistics on performance.
        self.summary = pd.DataFrame(index=range(self.generation_size))

        # Either breed previous gen or start fresh
        self.players = self.breed(breeders)

        self.gen_number = gen_number


    def breed(self, breeders=None):
        # Either breed generation from list of breeders or start fresh
        new_gen = []
        for i in range(self.generation_size):
            if breeders is not None:
                print('Breeding player {}.'.format(i))
                P1, P2 = np.random.choice(breeders, 2, replace=False)
                new_gen.append(P1.breed(P2))
            else:
                print('Creating player {} from scratch.'.format(i))
                new_gen.append(Player())

        return np.array(new_gen)

    def get_breeders(self):
        # Returns: Best performers based on self.scores
        performances = self.summary['performance']
        top_k_inds = np.argsort(performances)[::-1][:self.number_to_breed]
        return self.players[top_k_inds]

    def eval_players(self):
        # Have each player play the game and record performance
        # FIXME: This should be run in parallel
        scores = []
        durations = []
        for i, P in enumerate(self.players):

            print('Evaluating player {}..'.format(i))

            # Ugh, state.  This alters G in place.
            G = GameState()
            P.play_game(G)

            scores.append(G.score)
            durations.append(G.time)

        self.summary['score'] = scores
        self.summary['duration'] = durations
        self.summary['performance'] = (self.summary['score']*self.score_weight
                                       + self.summary['duration']*self.duration_weight)

    def advance_next_gen(self):
        # Updates self in place to form new generation.
        # Returns: self
        breeders = self.get_breeders()
        self.players = self.breed(breeders)

        # Metadata
        self.gen_number += 1
        self.summary = pd.DataFrame(index=range(self.generation_size))

        return self

    def save(self, save_dir='data'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        gen_dir = save_dir + '/gen' + str(self.gen_number)

        for i, P in enumerate(self.players):
            print('Saving player {0}...'.format(i))

            if not os.path.exists(gen_dir):
                os.mkdir(gen_dir)

            P.save_model_weights(gen_dir + '/player' + str(i) + '.h5')

        print('Saving summary.')
        self.summary.to_csv(save_dir + '/gen{0}_summary.csv'
                            .format(self.gen_number), index=False)

    def load(self, load_dir='data', use_gen=True):
        gen_dir = load_dir + '/gen' + str(self.gen_number)

        for i, P in enumerate(self.players):
            print('Loading model {0}...'.format(i))
            if use_gen:
                P.load_model_weights(gen_dir + '/player' + str(i) + '.h5')
            else:
                P.load_model_weights(load_dir + '/player' + str(i) + '.h5')

        print('Loading summary.')
        self.summary = pd.read_csv(load_dir + '/gen{0}_summary.csv'
                                   .format(self.gen_number))
