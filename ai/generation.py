import multiprocessing as mp
import os

import numpy as np
from numpy.random import choice, randint

from ai.player import Player
from config.init_config import InitConfig
from game.game_state import GameState
import pandas as pd


def _eval_iter(tup):
    i, weights = tup
    seed = randint(1000, 9999)
    print(f"Evaluating player {i} on game {seed}.")

    G = GameState(seed=seed)
    P = Player()
    P.model.set_weights(weights)
    P.play_game(G)

    return i, seed, G


class Generation(InitConfig):
    """
    This class is responsible for managing the spawning and ranking of one
    generation of player instances, as well as spawning its successor
    generation.
    """

    def __init__(self, breeders=None, gen_number=1, generation_size=None):
        super().__init__()

        # Dataframe with summary statistics on fitness.
        self.summary = None

        # Either breed previous gen or start fresh
        self.players = None

        # Metadata
        self.gen_number = gen_number
        # Seeds for random number generation.  Helps recreate games
        if generation_size is not None:
            self.generation_size = generation_size

    def breed(self, breeders=None):
        """
        Either breed generation from list of breeders or start fresh
        Returns: array of Player instances bred from breeders
        """
        new_gen = []
        for i in range(self.generation_size):

            if breeders is not None:
                # The breeders survive.
                if i < self.number_to_breed:
                    self._print("Persisting breeder %d." % i)
                    new_gen.append(breeders[i])
                else:
                    self._print("Breeding player %d." % i)
                    P1, P2 = choice(breeders, 2, replace=False)
                    new_gen.append(P1.breed(P2))

            else:
                self._print("Creating player %d from scratch." % i)
                new_gen.append(Player())

        return np.array(new_gen)

    def _print(self, msg):
        os.system("clear")
        print(msg)

    def spawn_random(self):
        """Cold start: Spawn the first generation of players."""
        players = self.breed()
        self.players = players

    def get_leader_board(self):
        """Returns: DataFrame of key metrics for top performers (breeders) only"""

        if self.summary is None:
            raise RuntimeError("Current gen has not been evaluated.")

        return (
            self.summary.sort_values("fitness", ascending=False)
            .head(self.number_to_breed)
            .reset_index()
        )

    def eval_players(self):
        """Have each player play the game and record performance"""
        new_summary = pd.DataFrame(
            columns=["model", "seed", "score", "duration", "fitness"]
        )

        self._print("Evaluating players.")
        with mp.get_context("spawn").Pool(mp.cpu_count()) as pool:
            results = pool.map(
                _eval_iter,
                [(i, P.model.get_weights()) for (i, P) in enumerate(self.players)],
            )

        for j, (i, seed, G) in enumerate(results):
            new_summary.loc[j] = (
                i,
                seed,
                G.score,
                G.duration,
                self.fitness_function(G.score, G.duration),
            )

        self.summary = new_summary

    def advance_next_gen(self):
        """
        Updates self in place to form new generation.
        Returns: self
        """
        leader_board = self.get_leader_board()
        breeders = self.players[[int(i) for i in leader_board["model"]]]
        players = self.breed(breeders)
        self.players = players

        # Metadata
        self.gen_number += 1

    def train_iter(self, num_loops=1):
        """
        Iterate over the standard breed-eval-save loop
        Requires at least one generation to be saved already.
        """
        if self.players is None:
            self._print("Loading latest generation and training %d more." % num_loops)
            self.load_latest_gen()

        for _ in range(num_loops):
            self._print("Advancing one generation.")
            self.advance_next_gen()

            self._print("Evaluating players...")
            self.eval_players()

            self._print("Saving generation.")
            self.save_latest_gen()

    def save_latest_gen(self):
        if self.players is None or self.summary is None:
            raise RuntimeError(
                "Need to breed or spawn players and evaluate" "before saving."
            )

        save_dir = "gen%04d" % self.gen_number
        for i, P in enumerate(self.players):
            self._print("Saving player %d..." % i)

            if not os.path.exists("data/" + save_dir):
                os.mkdir("data/" + save_dir)

            P.save_weights("data/%s/player%04d.h5" % (save_dir, i))

        self._print("Saving summary.")
        self.summary.to_csv("data/" + save_dir + "/summary.csv", index=False)

        self._print("Done.")

    def load_gen(self, gen_number):
        self._print("Loading generation %d." % gen_number)

        self._print("Loading summary.")
        self.summary = pd.read_csv(
            f"data/gen{gen_number:04.0f}/summary.csv", dtype={"seed": int}
        )

        load_dir = "gen%04d" % gen_number

        self.gen_number = gen_number

        players = []
        for i in range(self.generation_size):
            self._print("Loading model %d..." % i)
            P = Player()
            P.load_weights("data/%s/player%04d.h5" % (load_dir, i))
            players.append(P)

        self.players = np.array(players)

    def load_latest_gen(self):
        gens = [int(s[3:]) for s in os.listdir("data") if s.startswith("gen")]
        if len(gens) == 0:
            self.spawn_random()
            self.eval_players()
            self.save_latest_gen()
        else:
            gen_number = max(gens)
            self.load_gen(gen_number)
