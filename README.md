# snake-learning

I wanted to learn about genetic algorithms learning model, and while I was
brainstorming data collection, I threw this game together really quick.  The
neural network is a work in progress.

## Overview

### Goals

1. To get a neural network performing decently on a task.
2. Refresh my memory of how to use tesorflow and keras. (I've used them before,
   but never had much success with any of my projects.)
3. Enforce some python best practices I haven't been as rigorous about in the
   past.  These include writing unit tests and following standard project
   structure.

### How to play for yourself

To kick things off, I wrote a snake game to play.  Run `./play_snake.py` to try
it out.  Use your arrow keys to drive the snake.

### Credit where it's due

I am basically porting [this awesome repo] to python in order to understand
how the solution works.

[this awesome repo]: https://github.com/greerviau/SnakeAI

## Genetic algorithm

We're using a genetic algorithm.  That means that rather than optimizing weights
by training on lots of examples, we are simply randomly initializing a bunch of
players and letting them play.  Their performance is based on a scoring
function, which considers both the overall performace and the length of the game
played.  We take the top performers and "breed" them to make the next generation
of players.  This process is repeated until the performance stabilizes.

### Performance metric

When a player plays, the game state iterates a variable (call it `time`) each
time the snake moves (i.e., for each time the game state advances by one frame).
Each time the snake catches an apple, the game state iterates the score (call it
`score`).  The performance metric function for the game is

```python
performance = A*time + B*score
```

for some `A` and `B` which are tunable by the user.

### Breeding

Each generation, we spawn `num_spawn` players.  Of those, we take the top
`num_breed` performers based on the above metric, and breed pairs at random
with replacement to form `num_spawn` more players for the next generation.  To
breed a pair of top performers, we choose some of the weights from one and some
from the other.  We also apply Gaussian noise to introduce some randomness to
the child.  The variance of this noise can be thought of as a mutation rate, and
is configurable by the user.

#### Remark

The repo I'm borrowing from did this in a nonrandom fashion, by performing a
"crossover" operation for each layer.  In this operation, an index is selected
uniformly at random, all entries lexicographically before that index are drawn
from the left network, and all others are drawn from the right network.

I plan to try a few different breeding strategies.

## The neural network

We use a standard fully connected network with an input layer of size 18, two
hidden layers each of size 16, and an output of dimension 4.  The inputs encode
that the snake is looking in 8 directions (four cardinal and four diagonal).  In
each direction, she is looking for distance to an apple, distance to a wall, and
distance to herself.  The outputs encode which direction to move.
