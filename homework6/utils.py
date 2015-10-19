__author__ = 'hlin117'

import numpy as np

def make_deck(lands=24, spell1=10, spell2=10, spell3=10, spell4=2,
              spell5=5, spell6=6):
    """Creates a deck. The size of the deck varies depending on the
    number of each card. Lands are represented as 0. A spell of cost k is
    represented as k.

    Will raise a ValueError if the sum of the number of cards is
    not equal to 60.

    Returns
    -------
    A numpy array of size 60. The values of the cards are shuffled
    in this array.
    """
    decksize = lands + spell1 + spell2 + spell3 + spell4 + spell5 + spell6
    if decksize != 60:
        raise ValueError("The size of the deck does not equal 60.")

    # Using an absurd amount of operator overloading here
    deck = [0] * lands + [1] * spell1 + [2] * spell2 + [3] * spell3 \
           + [4] * spell4 + [5] * spell5 + [6] * spell6
    return np.random.permutation(np.array(deck))

def get_card_counts(hand):
    """Counts the number of occurances of each card. If a card
    has cost k, its count will appear in index k of the resulting
    array. In this case, the size of the return array is always 7,
    which the most expensive spell being cost 6.

    Remember that numpy arrays are zero indexed. Therefore, land
    cards (of cost zero) are placed in index zero, not 1.

    Returns
    -------
    A numpy array of length 7, corresponding to card counts.
    """

    # There's probably a numpy way of doing this without a
    # for loop.
    counts = np.zeros(7)
    for value in range(7):
        counts[value] = sum(hand == value)
    return counts

