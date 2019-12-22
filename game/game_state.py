import os
import pickle
import numpy as np

# My stuff
from config.init_config import InitConfig

class GameState(InitConfig):
    """
    This class is responsible for holding the state of the game at any given
    time.  Its main method is `update` which advances the game by one frame and
    updates the direction of the snake.  It also has a drawing method for fun.
    """
    def __init__(self):

        # Grab global config variables
        super().__init__()

        np.random.seed(self.seed)

        self.prize_loc = np.random.randint(0, self.board_size, 2)
        self.head_loc  = np.array([board_size // 2, board_size // 2])
        self.direction = np.array([0, 0]) # Represents the ordered pair (dy/dt, dx/dt)
        self.score     = 0
        self.time      = 0 # Keep track of how long the game has lasted
        self.dead      = False

        # The board will be drawn from this array. Positive values
        # are the snake's body, and negative values are the prizes.
        # At each update, all positive values greater than
        # self.score + 10 will be removed and a 1 will be placed adjacent
        # to the previous head of the snake.  This ensures that the snake
        # grows in length as more prizes are consumed.
        self.board = np.zeros((self.board_size, self.board_size))

        # Draw head and prize for the first frame.
        self.board[self.head_loc[0], self.head_loc[1]] = 1
        self.board[self.prize_loc[0], self.prize_loc[1]] = -1


    def update(self, new_direction):
        # Direction update (only if valid, i.e., no reversing direction)
        if not all(new_direction == -1*(self.direction)):
            self.direction = new_direction

        # Putative next location
        next_loc = self.head_loc + self.direction
        nextR, nextC = next_loc

        # Wall detection
        if nextC < 0 or nextC >= 20 or nextR < 0 or nextR >= 20:
            self.dead = True
            return

        # Self-collision detection
        if self.board[nextR, nextC] > 0:
            self.dead = True
            return

        # Prize handling
        if all(next_loc == self.prize_loc):
            self.score += 1
            X, Y = np.where(self.board == 0)
            i = np.random.choice(range(len(X)))
            self.prize_loc = np.array([X[i], Y[i]])

        # Update location
        self.head_loc += self.direction
        self.time += 1

        # Increment all nonzero cells (will also erase prize)
        self.board += (self.board != 0).astype(int)
        # Add new head cell
        self.board[self.head_loc[0], self.head_loc[1]] = 1
        # Delete tail cell
        self.board[self.board > self.score+10] = 0
        # Add prize cell
        self.board[self.prize_loc[0], self.prize_loc[1]] = -1


    def draw(self):
        os.system('clear')
        num_rows = self.board.shape[0]
        num_cols = self.board.shape[1]

        # This weird printing syntax is an old school way of placing text
        # at coordinates.  Unfortunately those coordinates are indexed from
        # one, so this code is pretty ugly.
        # Place a horizonal top border
        print('\033[1;1H+' + '-'*num_cols + '+')
        for i in range(2, num_rows+2):
            # Place part of the left border
            print('\033[{0};1H|'.format(i))
            for j in range(2, num_cols+2):
                # Print X for snake body, O for prize.
                if self.board[i-2, j-2] > 0:
                    print('\033[{0};{1}H{2}'.format(i, j, 'X'))
                elif self.board[i-2, j-2] < 0:
                    print('\033[{0};{1}H{2}'.format(i, j, 'O'))
            # Print part of the right border
            print('\033[{0};{1}H|'.format(i, num_cols+2))
        # Print bottom border
        print('\033[{0};0H+'.format(num_rows+2) + '-'*num_cols+ '+')
        # Print other self info
        print('\033[{0};1HScore: {1}, Head: ({2},{3}), Prize: ({4},{5})'
              .format(num_rows+3, self.score,
                      self.head_loc[0], self.head_loc[1],
                      self.prize_loc[0], self.prize_loc[1]))



    ###########################################################################
    # Methods for providing information to the player (maybe should be methods
    # of Player class.)
    ###########################################################################
    def scan_in_direction(self, direction):
        # TODO: Write tests!!!
        # Returns: triple of integers representing distance to edge, distance to
        # closest prize (if any), and distance to body (if any)
        r, c = self.head_loc

        def _reverse(arr):
            return arr[::-1]

        # TODO: Test these
        if direction == 'west':
            line_of_sight = _reverse(self.board[r, :c-1])
        elif direction == 'east':
            line_of_sight = self.board[r, c+1:]
        elif direction == 'north':
            line_of_sight = _reverse(self.board[:r-1, c])
        elif direction == 'south':
            line_of_sight = self.board[r+1:, c]
        elif direction == 'northwest':
            limit = min(r + 1, c + 1)
            rs = [r - i for i in range(1, limit)]
            cs = [c - i for i in range(1, limit)]
            line_of_sight = self.board[rs, cs]
        elif direction == 'northeast':
            limit = min(r + 1, self.board_size - c)
            rs = [r - i for i in range(1, limit)]
            cs = [c + i for i in range(1, limit)]
            line_of_sight = self.board[rs, cs]
        elif direction == 'southeast':
            limit = min(self.board_size - r, self.board_size - c)
            rs = [r + i for i in range(1, limit)]
            cs = [c + i for i in range(1, limit)]
            line_of_sight = self.board[rs, cs]
        elif direction == 'southwest':
            limit = min(self.board_size - r, c + 1)
            rs = [r + i for i in range(1, limit)]
            cs = [c - i for i in range(1, limit)]
            line_of_sight = self.board[rs, cs]
        else:
            raise RuntimeError('Invalid direction.')

        edge_distance = self._detect(line_of_sight != line_of_sight)
        prize_distance = self._detect(line_of_sight == -1)
        body_distance = self._detect(line_of_sight > 0)

        return edge_distance, prize_distance, body_distance

    def _detect(self, line_of_sight):
        # Expects: A numpy array of booleans
        # Returns: Closest true from 0

        # Null out chance of head detection (ugly!)
        line_of_sight[0] = False

        target_locs, = np.where(line_of_sight)
        if len(target_locs) == 0:
            # TODO: How to encode no target in sight?
            target_distance = len(line_of_sight)
        else:
            target_distance = target_locs[0]

        return target_distance

