import itertools

import numpy as np
import tqdm
from constants import DUDO, NUM_ACTIONS, NUM_SIDES
from history import get_active_player, get_next_history, is_terminal, last_action
from info_set import Node, action_num_to_ints, get_strategy, node_to_str
from numba import jit, typeof
from numba.core import types
from numba.typed import Dict, List
from numpy.typing import NDArray


@jit(nopython=True)
def payoff(dice: List, history):
    last_bid = last_action(history & ~(1 << DUDO))
    count, face = action_num_to_ints(last_bid)
    if count == 1:
        if dice[0] == face or dice[0] == 1 or dice[1] == face or dice[1] == 1:
            # Our opponent just called DUDO. But the last bid (ours) was correct,
            # so we win.
            return 1
        return -1
    elif (dice[0] == face or dice[0] == 1) and (dice[1] == face or dice[1] == 1):
        return 1
    return -1


@jit(nopython=True)
def cfr(
    dice: NDArray[np.int32],
    history: np.int64,
    p0: np.float32,
    p1: np.float32,
    node_map=Dict.empty(
        key_type=types.int64,
        value_type=typeof(Node(0)),
    ),
):
    if is_terminal(history):
        return payoff(dice, history)

    if p0 < 1e-10 and p1 < 1e-10:
        return 0.0
    active_player = get_active_player(history)
    info_set = dice[active_player] << 32 | history

    if info_set not in node_map:
        node_map[info_set] = Node(info_set)
    node = node_map[info_set]

    node_util = 0.0
    util = np.zeros(NUM_ACTIONS, dtype=np.float32)

    last_bid_action = last_action(history)
    strategy = get_strategy(node.regret_sum)
    for a in range(last_bid_action + 1, NUM_ACTIONS):
        if last_bid_action == -1 and a == DUDO:
            continue

        next_history = get_next_history(history, a)
        if active_player == 0:
            util[a] = -cfr(dice, next_history, p0 * strategy[a], p1, node_map)
        else:
            util[a] = -cfr(dice, next_history, p0, p1 * strategy[a], node_map)

        node_util += strategy[a] * util[a]

    for a in range(last_bid_action + 1, NUM_ACTIONS):
        if last_bid_action == -1 and a == DUDO:
            continue
        regret = util[a] - node_util
        node.regret_sum[a] += (p1 if active_player == 0 else p0) * regret
        node.strategy_sum[a] += strategy[a] * (p0 if active_player == 0 else p1)

    return node_util


def train(iterations, print_nodes=True):
    util = np.zeros(iterations, dtype=np.float32)

    node_map = Dict.empty(
        key_type=types.int64,
        value_type=typeof(Node(0)),
    )
    it = itertools.cycle(itertools.product(range(1, NUM_SIDES + 1), repeat=2))
    for i in tqdm.tqdm(range(iterations)):
        dice = List(next(it))
        run_util = cfr(dice, 0, 1.0, 1.0, node_map)
        util[i] = run_util

    if print_nodes:
        for node in node_map.values():
            print(node_to_str(node))
        avg = util.mean()
        print(f"Average game value: {avg}")
    return util
