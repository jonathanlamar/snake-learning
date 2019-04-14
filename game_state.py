import os
import pickle
import numpy as np

class GameState:
    def __init__(self, seed=None, board_size=20):
        np.random.seed(seed)

        self.board_size = board_size
        self.prize_loc = np.random.randint(0, self.board_size, 2)
        self.head_loc  = np.array([board_size // 2, board_size // 2])
        self.direction = np.array([0, 0])
        self.score     = 0
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
        self.head_loc += self.direction

        # Wall detection
        if any(self.head_loc >= 20) or any(self.head_loc < 0):
            self.dead = True
            return

        # Self-collision detection
        if self.board[self.head_loc[0], self.head_loc[1]] > 0:
            self.dead = True
            return

        # Prize handling
        if all(self.head_loc == self.prize_loc):
            self.score += 1
            X, Y = np.where(self.board == 0)
            i = np.random.choice(range(len(X)))
            self.prize_loc = np.array([X[i], Y[i]])

        # Increment all nonzero cells (will also erase prize)
        self.board += (self.board != 0).astype(int)
        # Add new head cell
        self.board[self.head_loc[0], self.head_loc[1]] = 1
        # Delete tail cell
        self.board[self.board > self.score+10] = 0
        # Add prize cell
        self.board[self.prize_loc[0], self.prize_loc[1]] = -1

    def output_state(self, time):
        visible_state = self.board.copy()
        # The positive values are incremented, so we  need to reset them to 1.
        visible_state[visible_state > 0] = 1
        # TODO: Implement this.

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
