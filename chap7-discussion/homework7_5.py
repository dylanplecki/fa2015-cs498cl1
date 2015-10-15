#!/usr/bin/env python
from __future__ import division
import numpy as np
from utils import make_deck, get_card_counts
from time import time

__author__ = 'hlin117'

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
    for spell_value in range(lands_on_table, -1, -1):
        if card_counts[spell_value] > 0:
            card_counts[spell_value] -= 1
            return spell_value, lands_on_table
    return -1, lands_on_table

"""Simulates the experiment done in listing 7.2 of the textbook:

"Assume you have a deck of

- 24 lands
- 10 spells of cost 1
- 10 spells of cost 2
- 10 spells of cost 3
- 2 spells of cost 4
- 2 spells of cost 5
- 2 spells of cost 6

What is the probability you will be able to play at least one
spell on each of the four turns? (Page 210.)"

The code is on page 216.
"""

start = time()
n_experiments = 10  # He calls these "simulations"
n_simulations = 1000  # He calls these "inner simulations"

# 6 x 6 table of zeros. The first dimension (row) represents the
# turn, indexed at 0. The second dimension (col) represents the
# "spell value - 1". (For example, index 5 corresponds to a spell value of 6.)
counts = np.zeros((6, 6))

for sim in range(n_simulations):

    deck = make_deck(lands=24, spell1=10, spell2=10, spell3=10, spell4=2,
                     spell5=2, spell6=2)

    hand = deck[:7]
    card_counts = get_card_counts(hand)

    # From here, it's a matter of playing the game. Limit ourselves
    # to only 4 turns.
    n_turns = 6
    spells_played = list()
    lands_on_table = 0
    next_card_ind = 7
    for turn_ind in range(n_turns):
        spell_value, lands_on_table \
            = play_round(lands_on_table, next_card_ind)
        next_card_ind += 1

        if spell_value > 0:
            spell_ind = spell_value - 1
            counts[turn_ind][spell_ind] += 1
            break

probabilities = counts / n_simulations
n_seconds = round(time() - start)
print("Probabilities: ".ljust(25) + str(probabilities))
print("Mean: ".ljust(25) + str(round(probabilities.mean(), 4)))
print("Standard deviation: ".ljust(25) + str(round(probabilities.std(), 4)))
print("Number of seconds: ".ljust(25) + str(format(n_seconds)))

