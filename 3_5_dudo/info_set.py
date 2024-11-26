import numpy as np
from constants import DUDO, NUM_ACTIONS, NUM_SIDES
from numba import float32, jit
from numba.core import types
from numba.experimental import jitclass

from history import get_active_player


@jitclass(
    [
        ("info_set", types.int64),
        ("regret_sum", float32[:]),
        ("strategy_sum", float32[:]),
    ]
)
class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        self.regret_sum = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.strategy_sum = np.zeros(NUM_ACTIONS, dtype=np.float32)


@jit(nopython=True)
def get_strategy(regret_sum: np.ndarray) -> np.ndarray:
    strategy = np.maximum(regret_sum, np.float32(0.0))
    normalizing_sum = strategy.sum(dtype=np.float32)
    if normalizing_sum > 0:
        strategy /= normalizing_sum
    else:
        strategy = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)

    return strategy


@jit(nopython=True)
def get_avg_strategy(strategy_sum: np.ndarray):
    avg_strategy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    normalizing_sum = 0.0
    for i in range(NUM_ACTIONS):
        normalizing_sum += strategy_sum[i]
    for i in range(NUM_ACTIONS):
        if normalizing_sum > 0:
            avg_strategy[i] = strategy_sum[i] / normalizing_sum
        else:
            avg_strategy[i] = 1 / NUM_ACTIONS
    return avg_strategy


def node_to_str(node):
    avg_strat = get_avg_strategy(node.strategy_sum)
    info_set = node.info_set

    s = "["
    for i in range(NUM_ACTIONS):
        if avg_strat[i] > 0.001:
            s += f"{action_num_to_str(i)}:{avg_strat[i]:.3f},"
    s = s[:-1] + "]"
    return f"{hex(info_set)} / {info_set_to_str(info_set)}: {s}"


@jit(nopython=True)
def action_num_to_str(action_num):
    if action_num == DUDO:
        return "DUDO"
    count, face = action_num_to_ints(action_num)
    return f"{count}x{face}"


@jit(nopython=True)
def action_num_to_ints(action_num):
    return action_num // NUM_SIDES + 1, (action_num + 1) % NUM_SIDES + 1


@jit(nopython=True)
def info_set_to_str(info_set):
    dice = info_set >> 32
    history = info_set & 0xFFFFFFFF

    active_player = get_active_player(history)

    s = f"({active_player}) "
    for a in range(NUM_ACTIONS):
        if history & (1 << a):
            s += action_num_to_str(a) + ","
    return f"{dice} {s[:-1]}"
