import numpy as np
from itertools import product

# My stuff
from config.init_config import InitConfig
from game.game_state import GameState

# Tests for making sure the update method works as expected.
def test_north_collision():
    G = GameState()

    # Initialize game state
    G.head_loc = np.array([0, 0])
    G.direction = np.array([0, 1])

    # Going right along the north wall (make sure it's possible)
    G.update(new_direction=np.array([0, 1]))
    assert not G.dead

    # Turn and collide
    G.update(new_direction=np.array([-1, 0]))
    assert G.dead

def test_south_collision():
    G = GameState()

    # Initialize game state
    G.head_loc = np.array([G.board_size-1, 0])
    G.direction = np.array([0, 1])

    # Going right along the south wall (make sure it's possible)
    G.update(new_direction=np.array([0, 1]))
    assert not G.dead

    # Turn and collide
    G.update(new_direction=np.array([1, 0]))
    assert G.dead

def test_east_collision():
    G = GameState()

    # Initialize game state
    G.head_loc = np.array([0, G.board_size-1])
    G.direction = np.array([1, 0])

    # Going down along the east wall (make sure it's possible)
    G.update(new_direction=np.array([1, 0]))
    assert not G.dead

    # Turn and collide
    G.update(new_direction=np.array([0, 1]))
    assert G.dead

def test_west_collision():
    G = GameState()

    # Initialize game state
    G.head_loc = np.array([0, 0])
    G.direction = np.array([1, 0])

    # Going down along the west wall (make sure it's possible)
    G.update(new_direction=np.array([1, 0]))
    assert not G.dead

    # Turn and collide
    G.update(new_direction=np.array([0, -1]))
    assert G.dead


def test_body_collision():
    G = GameState()
    bs = G.board_size
    hl = G.head_loc

    # Forge game state right before body collision
    arr = np.zeros((bs, bs))
    arr[hl[0]::-1, hl[1]+1] = 1
    G.board = arr

    # Going up alongside fake body
    G.update(new_direction=np.array([-1,0]))
    assert not G.dead

    # Turning in to the body for collision
    G.update(new_direction=np.array([0, 1]))
    assert G.dead

def test_prize_collection():
    G = GameState()
    hl = G.head_loc

    # Forge game state right before prize collection
    G.board[np.where(G.board == -1)] = 0
    G.board[hl[0]-1, hl[1]] = -1
    G.prize_loc = np.array([hl[0]-1, hl[1]])

    # Going up alongside fake body
    G.update(new_direction=np.array([-1,0]))
    assert G.score == 1

# Tests for LOS generation
def test_LOS_generation():
    # TODO: Write tests to check order and length.
    config = InitConfig()
    bs = config.board_size

    for _ in range(100):
        G = GameState()

        # Randomize head loc
        G.board = np.zeros((bs, bs))
        G.head_loc = np.random.randint(0, bs, 2)
        G.update(new_direction=np.array([0,0]))

        for dy, dx in product([-1, 0, 1], [-1, 0, 1]):
            if dy == 0 and dx == 0:
                continue

            LOS = G.get_line_of_sight(dy, dx)

            # LOS does not contain head
            assert np.all(LOS != 1)
