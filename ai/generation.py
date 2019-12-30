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

        # Dataframe with summary statistics on fitness.
        self.summary = None

        # Either breed previous gen or start fresh
        self.players = None

        # Metadata
        self.gen_number = gen_number
        # Seeds for random number generation.  Helps recreate games
        if generation_size is not None:
            self.generation_size=generation_size


    def breed(self, breeders=None):
        # Either breed generation from list of breeders or start fresh
        # Returns: array of Player instances bred from breeders

        new_gen = []
        for i in range(self.generation_size):

            if breeders is not None:
                # The top half of the breeders survive.
                if i < self.number_to_breed:
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
        # Cold start: Spawn the first generation of players.
        players = self.breed()
        self.players = players


    def get_leader_board(self):
        # Returns: DataFrame of key metrics for top performers (breeders) only

        if self.summary is None:
            raise RuntimeError('Current gen has not been evaluated.')

        return (self.summary
                .query('generation == %d' % self.gen_number)
                .drop('generation', axis=1)
                .sort_values('fitness', ascending=False)
                .head(self.number_to_breed))


    def eval_players(self):
        # Have each player play the game and record performance
        scores    = np.zeros(self.generation_size)
        durations = np.zeros(self.generation_size)
        perfs     = np.zeros(self.generation_size)
        seeds     = np.zeros(self.generation_size, dtype=int)

        for i, P in enumerate(self.players):

            print('Evaluating player %d..' % i)

            # Ugh, state.  This alters G in place.
            seed = randint(1000, 9999)
            G = GameState(seed=seed)
            P.play_game(G)

            seeds[i]     = seed
            scores[i]    = G.score
            durations[i] = G.duration
            perfs[i]     = self.fitness_function(G.score, G.duration)

        new_summary = pd.DataFrame({
            'generation' : self.gen_number,
            'model'      : range(self.generation_size),
            'seed'       : seeds,
            'score'      : scores,
            'duration'   : durations,
            'fitness'    : perfs
        })

        # Overwrite previous eval if exists
        if self.summary is None:
            self.summary = new_summary
        else:
            df = self.summary[self.summary['generation'] < self.gen_number]
            self.summary = pd.concat([df, new_summary])


    def show_best(self):
        # For checking qualitative behavior of best performers
        leader_board = self.get_leader_board()

        players = self.players[leader_board['model']]
        seeds = leader_board['seed'].astype(int)

        for P, seed in zip(players, seeds):
            G = GameState(seed=seed)
            P.play_game(G, draw_game=True)


    def reevaluate_best_players(self, games_per_player=10):
        leader_board = self.get_leader_board()
        inds = leader_board['model']

        scores    = []
        durations = []
        perfs     = []
        players   = []

        for ind, P in zip(inds, self.players[inds]):
            print('Evaluating player %d on %d games.' % (ind, games_per_player))
            for i in range(games_per_player):
                print('Game %d...' % (i + 1))

                G = GameState()
                P.play_game(G)

                players.append(ind)
                scores.append(G.score)
                durations.append(G.duration)
                perfs.append(self.fitness_function(G.score, G.duration))

        df = (pd
              .DataFrame({
                  'model'       : players,
                  'score'       : scores,
                  'duration'    : durations,
                  'fitness' : perfs
              })
              .groupby('model')
              .agg(
                  mean_score   = ('score', np.mean),
                  median_score = ('score', np.median),
                  max_score    = ('score', np.max),
                  std_score    = ('score', np.std),

                  mean_dur     = ('duration', np.mean),
                  median_dur   = ('duration', np.median),
                  max_dur      = ('duration', np.max),
                  std_dur      = ('duration', np.std),

                  mean_perf    = ('fitness', np.mean),
                  median_perf  = ('fitness', np.median),
                  max_perf     = ('fitness', np.max),
                  std_perf     = ('fitness', np.std)
              ))

        return df


    def get_best_player(self):
        leader_board = self.get_leader_board()
        best_ind = leader_board['model'].iloc[0]
        return self.players[best_ind]


    def advance_next_gen(self):
        # Updates self in place to form new generation.
        # Returns: self
        leader_board = self.get_leader_board()
        breeders     = self.players[leader_board['model']]
        players      = self.breed(breeders)
        self.players = players

        # Metadata
        self.gen_number += 1


    def train_iter(self, num_loops=1):
        # Iterate over the standard breed-eval-save loop
        # Requires at least one generation to be saved already.
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

        if self.players is None or self.summary is None:
            raise RuntimeError('Need to breed or spawn players and evaluate'
                               'before saving.')

        save_dir = 'test' if test else 'gen%04d' % self.gen_number

        for i, P in enumerate(self.players):
            print('Saving player %d...' % i)

            if not os.path.exists('data/' + save_dir):
                os.mkdir('data/' + save_dir)

            P.save_weights('data/%s/player%04d.h5' % (save_dir, i))

        print('Saving summary.')
        df_name = 'summary_test.csv' if test else 'summary.csv'
        self.summary.to_csv('data/' + df_name, index=False)


    def load_gen(self, gen_number, test=False):
        print('Loading generation %d.' % gen_number)

        print('Loading summary.')
        df_name = 'summary_test.csv' if test else 'summary.csv'
        df = pd.read_csv(
            'data/' + df_name,
            dtype={'seed' : int}
        )
        df = df[df['generation'] <= gen_number]

        load_dir = 'test' if test else 'gen%04d' % gen_number

        self.summary = df
        self.gen_number = gen_number

        players = []
        for i in range(self.generation_size):
            print('Loading model %d...' % i)
            P = Player()
            P.load_weights('data/%s/player%04d.h5' % (load_dir, i))
            players.append(P)

        self.players = np.array(players)

    def load_latest_gen(self, test=False):
        df_name = 'summary_test.csv' if test else 'summary.csv'
        df = pd.read_csv('data/' + df_name)
        gen_number = df['generation'].max()

        self.load_gen(gen_number, test)
