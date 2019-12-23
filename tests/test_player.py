import numpy as np

# My stuff
from ai.player import Player
from config.init_config import InitConfig

def test_cross_singleton():
    arr1 = np.zeros((1, 1))
    arr2 = np.ones((1, 1))

    player = Player()
    cross = player._cross_arrays(arr1, arr2)

    assert cross == arr1 or cross == arr2

def test_rowsums():
    player = Player()
    bs = player.board_size

    arr1 = np.zeros((bs, bs))
    arr2 = np.ones((bs, bs))

    for _ in range(100):
        cross = player._cross_arrays(arr1, arr2)

        rowsums = cross.sum(axis=1)
        # rowsum( 0 0 0
        #         0 1 1
        #         1 1 1) == 3
        # rowsum( 0 0 0
        #         1 1 1
        #         1 1 1) == 2
        # rowsum( 1 1 1
        #         1 1 1
        #         1 1 1) == 1
        assert len(set(rowsums)) in [1, 2, 3]

def test_colsums():
    player = Player()
    bs = player.board_size

    arr1 = np.zeros((bs, bs))
    arr2 = np.ones((bs, bs))

    for _ in range(100):
        cross = player._cross_arrays(arr1, arr2)

        colsums = cross.sum(axis=0)
        # colsum( 0 0 0
        #         0 1 1
        #         1 1 1) == 2
        # colsum( 0 0 0
        #         1 1 1
        #         1 1 1) == 1
        assert len(set(colsums)) in [1, 2]

def test_fuzz_mean():
    arr = np.zeros(2000000)

    player = Player()
    arr_fuzz = player._mutate_array(arr)

    # This will fail with nonzero probability
    assert np.abs(arr_fuzz.mean()) < 0.002

def test_fuzz_var():
    arr = np.zeros(2000000)

    player = Player()
    arr_fuzz = player._mutate_array(arr)

    # This will fail with nonzero probability
    config = InitConfig()
    assert np.abs(arr_fuzz.var() - config.mutation_rate) < 0.01
