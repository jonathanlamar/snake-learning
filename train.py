import sys

from ai.generation import Generation

if __name__ == "__main__":
    num_gens = int(sys.argv[1])

    gen = Generation()
    gen.load_latest_gen()
    gen.train_iter(num_gens)

    print(f"Done training {num_gens} generations.\nFinal Leaderboard:")
    print(gen.get_leader_board())
