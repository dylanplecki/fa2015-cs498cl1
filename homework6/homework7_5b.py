#!/usr/bin/env python
from __future__ import division
import numpy as np
from utils import make_deck, get_card_counts
from time import time

__author__ = 'Henry Lin <halin2@illinois.edu>'

def play_round(lands_on_table, next_card_ind):
    """Plays the round. Notice that this function is a function inside
    a function, so it could modify the variables outside of its environment.
    This is called a closure.

    In this case, we're using the following outside variables:

    - deck
    - card_counts

    This is how I could have less input parameters than David's code.

    (Immutable objects in python, such as ints, cannot be access from
    outside the function. Unless you're working with python3; there is
    the "nonlocal" keyword.)

    **NOTE**: I was about to actually make this a non-closure, because
    having a closure is only going to cause confusion. Too late for that
    now...

    Returns
    -------
    played_spell : -1 if no spell was played. Otherwise, returns the value
    of the spell played.

    lands_on_table : The number of lands on the table
    """

    assert deck is not None, "Need to define a numpy array representing" \
                             "the deck, called 'deck'."
    assert card_counts is not None, "Need to define a numpy array representing" \
                                    "the counts of each card"

    next_card = deck[next_card_ind]
    card_counts[next_card] += 1

    # Put a land on the table if we have one in the hand
    if card_counts[0] > 0:
        lands_on_table += 1
        card_counts[0] -= 1

    # In this selection process, we are interested in picking the
    # most expensive card per round. (Yes, the range loop below is right.)
    for spell_value in range(min(lands_on_table, 6), 0, -1):
        if card_counts[spell_value] > 0:
            card_counts[spell_value] -= 1
            return spell_value, lands_on_table
    return -1, lands_on_table

"""For problem 7.5 in the textbook.

"Assume you have a deck of

- 24 lands
- 10 spells of cost 1
- 10 spells of cost 2
- 10 spells of cost 3
- 2 spells of cost 4
- 2 spells of cost 5
- 2 spells of cost 6

It is properly shuffled, and you draw seven cards. At each turn,
you play a land if it is in your hand, and you always only play the
most expensive spell in your hand that you area able to play and you
never play two spells.

Part b: Write a program to estimate the probability that the first spell you play
has cost 4."
"""

start = time()
n_experiments = 10  # He calls these "simulations"
n_simulations = 10000  # He calls these "inner simulations"
counts = np.zeros(n_experiments)

for exp in range(n_experiments):
    for sim in range(n_simulations):

        deck = make_deck(lands=24, spell1=10, spell2=10, spell3=10, spell4=2,
                         spell5=2, spell6=2)

        hand = deck[:7]
        card_counts = get_card_counts(hand)

        # From here, it's a matter of playing the game
        lands_on_table = 0
        next_card_ind = 7
        while True:
            spell_value, lands_on_table \
                    = play_round(lands_on_table, next_card_ind)
            next_card_ind += 1

            # TODO: CHANGE ME FOR YOUR EXPERIMENT! (If needed...)
            # What do I do with the spell value?
            # You might want to use the spell_value that you played, above.
            if spell_value > 0:
                if spell_value == 4:
                    counts[exp] += 1
                break

probabilities = counts / n_simulations
n_seconds = round(time() - start)
print("Probabilities: ".ljust(25) + str(probabilities))
print("Mean: ".ljust(25) + str(round(probabilities.mean(), 4)))
print("Standard deviation: ".ljust(25) + str(round(probabilities.std(), 4)))
print("Number of seconds: ".ljust(25) + str(format(n_seconds)))

