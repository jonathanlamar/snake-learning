import numpy as np
from numpy.random import choice, randint
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
    def __init__(self, breeders=None, gen_number=1, generation_size=None):

        # Grab global config variables
        super().__init__()

        # Dataframe with summary statistics on performance.
        self.summary = pd.DataFrame()

        # Either breed previous gen or start fresh
        self.players = None

        # Metadata
        self.gen_number = gen_number
        # Seeds for random number generation.  Helps recreate games
        # TODO: Write test
        if generation_size is not None:
            self.generation_size=generation_size


    def breed(self, breeders=None):
        # Either breed generation from list of breeders or start fresh
        new_gen = []
        for i in range(self.generation_size):
            if breeders is not None:
                # The top half of the breeders survive.
                if i < self.number_to_breed // 2:
                    print('Persisting breeder %d.' % i)
                    new_gen.append(breeders[i])
                else:
                    print('Breeding player %d.' % i)
                    P1, P2 = choice(breeders, 2, replace=False)
                    new_gen.append(P1.breed(P2))
            else:
                print('Creating player %d from scratch.' % i)
                new_gen.append(Player())

        return np.array(new_gen)

    def spawn_random(self):
        players = self.breed()
        self.players = players

    def get_leader_board(self):
        return (self.summary.loc[
                   self.summary['generation'] == self.gen_number,
                   ['model', 'seed', 'performance']]
                .sort_values('performance', ascending=False)
                .head(self.number_to_breed))

    def eval_players(self):
        # Have each player play the game and record performance
        # FIXME: This should be run in parallel
        scores = np.zeros(self.generation_size)
        durations = np.zeros(self.generation_size)
        perfs = np.zeros(self.generation_size)
        seeds = np.zeros(self.generation_size)
        for i, P in enumerate(self.players):

            print('Evaluating player %d..' % i)

            # Ugh, state.  This alters G in place.
            seed = randint(1000, 9999)
            G = GameState(seed=seed)
            P.play_game(G)

            seeds[i] = seed
            scores[i] = G.score
            durations[i] = G.time
            perfs[i] = self._get_performance(G.score, G.time)

        new_summary = pd.DataFrame({
            'generation' : self.gen_number,
            'model' : range(self.generation_size),
            'seed' : seeds,
            'score' : scores,
            'duration' : durations,
            'performance' : perfs
        })

        # Overwrite previous eval if exists
        if self.summary.shape[0] == 0:
            self.summary = new_summary
        else:
            df = self.summary[self.summary['generation'] < self.gen_number]
            self.summary = pd.concat([df, new_summary])

    def _get_performance(self, score, duration):
        # return score * self.score_weight + duration * self.duration_weight
        # TODO: This will start to fail eventually
        if score < 10:
            perf = (duration**2) * (2**score)
        else:
            perf = (duration**2) * (2**10) * (score - 9)

        return perf

    def show_best(self):
        leader_board = self.get_leader_board()

        # TODO: Debug this
        players = self.players[leader_board['model']]
        seeds = leader_board['seed'].astype(int)
        scores = leader_board['performance']

        for P, seed, score in zip(players, seeds, scores):
            G = GameState(seed=seed)
            P.play_game(G, draw_game=True)
            assert score == self._get_performance(G.score, G.time)

    def advance_next_gen(self):
        # Updates self in place to form new generation.
        # Returns: self
        leader_board = self.get_leader_board()
        breeders = self.players[leader_board['model']]
        players = self.breed(breeders)
        self.players = players

        # Metadata
        self.gen_number += 1

    def train_iter(self, num_loops=1):
        if self.players is None:
            print('Loading latest generation and training %d more.' % num_loops)
            self.load_latest_gen()

        for _ in range(num_loops):
            print('Advancing one generation.')
            self.advance_next_gen()

            print('Evaluating players...')
            self.eval_players()

            print('Saving generation.')
            self.save_latest_gen()

    def save_latest_gen(self, test=False):
        save_dir = 'test' if test else 'gen%04d' % self.gen_number

        for i, P in enumerate(self.players):
            print('Saving player %d...' % i)

            if not os.path.exists('data/' + save_dir):
                os.mkdir('data/' + save_dir)

            P.save_weights('data/%s/player%04d.h5' % (save_dir, i))

        print('Saving summary.')
        df_name = 'summary_test.csv' if test else 'summary.csv'
        self.summary.to_csv('data/' + df_name, index=False)

    def load_latest_gen(self, test=False):

        print('Loading summary.')
        df_name = 'summary_test.csv' if test else 'summary.csv'
        df = pd.read_csv('data/' + df_name)
        gen_number = df['generation'].max()
        self.summary = df
        self.gen_number = gen_number
        load_dir = 'test' if test else 'gen%04d' % self.gen_number

        players = []
        for i in range(self.generation_size):
            print('Loading model %d...' % i)
            P = Player()
            P.load_weights('data/%s/player%04d.h5' % (load_dir, i))
            players.append(P)

        self.players = np.array(players)
