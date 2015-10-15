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

def play_round(lands_on_table, next_card_ind):
    """Plays the round. Notice that this function is a function
    inside a function, so it could modify the variables outside
    of its environment. This is called a closure.

    In this case, we're using the following outside variables:

    - deck
    - card_counts

    This is how I could have less input parameters than David's code.

    (Immutable objects in python, such as ints, cannot be access from
    outside the function. Unless you're working with python3; there is
    the "nonlocal" keyword.)

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
    # card of least cost per round. If lands_on_table == 0, we don't
    # enter the for loop.
    for spell_value in range(1, lands_on_table + 1):
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

n_experiments = 10  # He calls these "simulations"
n_simulations = 1000  # He calls these "inner simulations"
counts = np.zeros(n_experiments)

for exp in range(n_experiments):
    for sim in range(n_simulations):

        deck = make_deck(lands=24, spell1=10, spell2=10, spell3=10, spell4=2,
                         spell5=2, spell6=2)

        hand = deck[:7]
        card_counts = get_card_counts(hand)

        # From here, it's a matter of playing the game. Limit ourselves
        # to only 4 turns.
        n_turns = 4
        spells_played = list()
        lands_on_table = 0
        next_card_ind = 7
        for turn in range(n_turns):
            spell_value, lands_on_table \
                    = play_round(lands_on_table, next_card_ind)
            next_card_ind += 1
            spells_played.append(spell_value)

        # Now we check that we played a card on each of the turns.
        # The boolean converts to an integer, because counts is a
        # (numpy) array of integers.
        bleh = int(all(spell_value > 0 for spell_value in spells_played))
        counts[exp] += bleh
print counts / n_simulations

