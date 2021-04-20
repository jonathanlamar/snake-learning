import numpy as np
from time import sleep
from pynput.keyboard import Key, Listener
from game.game_state import GameState

def on_press(key):
    global DIRECTION
    if key == Key.up:
        DIRECTION = np.array([-1, 0])
    elif key == Key.down:
        DIRECTION = np.array([1, 0])
    elif key == Key.left:
        DIRECTION = np.array([0, -1])
    elif key == Key.right:
        DIRECTION = np.array([0, 1])

def main_loop(game):
    global DIRECTION
    DIRECTION = np.array([-1, 0]) # Going up initially

    while game.dead == False:

        # Update game state
        game.update(DIRECTION)

        # Draw to screen
        game.draw()

        # Control the speed of the game here
        sleep(0.15)

if __name__ == '__main__':
    with Listener(on_press=on_press) as listener:
        game = GameState()
        main_loop(game)
    print('You lose!')
