#!/usr/bin/env python3

# NOTE: You must spawn one generation, eval, and save before this script can be
# run.  This will breed the next generation, eval, and save.

import sys

from ai.generation import Generation

if __name__ == "__main__":
    num_gens = int(sys.argv[1])

    gen = Generation()
    gen.load_latest_gen()
    gen.train_iter(num_gens)
