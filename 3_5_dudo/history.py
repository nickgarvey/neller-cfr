from constants import DUDO, NUM_ACTIONS
from numba import jit


@jit(nopython=True)
def get_active_player(history):
    total_bits = 0
    for i in range(NUM_ACTIONS):
        total_bits += 1 if history & (1 << i) else 0
    return total_bits % 2


@jit(nopython=True)
def get_next_history(history, action):
    return history | (1 << action)


@jit(nopython=True)
def is_terminal(history):
    return history & (1 << DUDO) != 0

@jit(nopython=True)
def last_action(history):
    for i in range(NUM_ACTIONS - 1, -1, -1):
        if history & (1 << i):
            return i
    return -1
