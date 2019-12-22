import numpy as np
from player import Player
from init_config import InitConfig

def test_cross_singleton():
    arr1 = np.zeros((1, 1))
    arr2 = np.ones((1, 1))

    player = Player()
    cross = player._cross_arrays(arr1, arr2)

    assert cross == arr1 or cross == arr2

def test_rowsums():
    arr1 = np.zeros((20, 20))
    arr2 = np.ones((20, 20))

    player = Player()
    cross = player._cross_arrays(arr1, arr2)

    rowsums = cross.sum(axis=1)

    assert len(set(rowsums)) == 3

def test_colsums():
    arr1 = np.zeros((20, 20))
    arr2 = np.ones((20, 20))

    player = Player()
    cross = player._cross_arrays(arr1, arr2)

    colsums = cross.sum(axis=0)

    assert len(set(colsums)) == 2

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
