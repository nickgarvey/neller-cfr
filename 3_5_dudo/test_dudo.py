from constants import DUDO, NUM_ACTIONS
from dudo import cfr, payoff
from history import get_active_player, get_next_history, is_terminal, last_action
from info_set import (
    Node,
    action_num_to_str,
    get_avg_strategy,
    get_strategy,
    node_to_str,
)
from numba import typeof
from numba.core import types
from numba.typed import Dict, List


def test_last_action():
    history = 0
    assert last_action(history) == -1

    history = get_next_history(history, 4)
    assert last_action(history) == 4

    history = get_next_history(history, DUDO)
    assert last_action(history) == DUDO


def test_is_terminal():
    history = 0
    assert not is_terminal(history)

    history = get_next_history(history, 4)
    assert not is_terminal(history)

    history = get_next_history(history, DUDO)
    assert is_terminal(history)


def test_get_active_player():
    history = 0
    assert get_active_player(history) == 0

    history = get_next_history(history, 4)
    assert get_active_player(history) == 1

    history = get_next_history(history, 5)
    assert get_active_player(history) == 0

    history = get_next_history(history, 10)
    assert get_active_player(history) == 1


def test_info_set_to_str():
    test_map = {
        0: "1x2",
        1: "1x3",
        2: "1x4",
        3: "1x5",
        4: "1x6",
        5: "1x1",
        6: "2x2",
        7: "2x3",
        NUM_ACTIONS - 2: "2x1",
        NUM_ACTIONS - 1: "DUDO",
    }

    for k, v in test_map.items():
        assert action_num_to_str(k) == v, (k, v, action_num_to_str(k))


def build_history(actions):
    history = 0
    for action in actions:
        history = get_next_history(history, action)
    return history


def last_bid_str(history):
    return action_num_to_str(last_action(history & ~(1 << DUDO)))


def test_payoff():
    # 2, 2
    dice = List([2, 2])
    # 1x2, 1x3
    history = build_history([0, 1, DUDO])
    assert payoff(dice, history) == -1, last_bid_str(history)

    # 1, 6
    dice = List([1, 6])
    # 1x2, 2x6
    history = build_history([1, DUDO - 2, DUDO])
    assert payoff(dice, history) == 1, last_bid_str(history)

    # 6, 6
    dice = List([6, 6])
    # 1x2, 2x6
    history = build_history([1, DUDO - 2, DUDO])
    assert payoff(dice, history) == 1, last_bid_str(history)

    # 6, 6
    dice = List([2, 2])
    # 1x3
    history = build_history([1, DUDO])
    assert payoff(dice, history) == -1, last_bid_str(history)


def test_get_avg_strategy():
    eps = 0.0001

    strat_sum = List([1.0] * NUM_ACTIONS)
    avg_strat = get_avg_strategy(strat_sum)
    assert 1 / NUM_ACTIONS - eps < avg_strat[DUDO] < 1 / NUM_ACTIONS + eps, avg_strat

    strat_sum = List([1.0] * NUM_ACTIONS)
    strat_sum[10] = 10000
    avg_strat = get_avg_strategy(strat_sum)
    assert avg_strat[10] > 0.99, avg_strat


def test_cfr():
    dice = List([6, 6])
    node_map = Dict.empty(
        key_type=types.int64,
        value_type=typeof(Node(0)),
    )
    for _ in range(10):
        cfr(dice, 0, 1.0, 1.0, node_map)

    strategy = get_strategy(node_map[0x6000003C0].regret_sum)
    assert strategy[DUDO] > 0.95, node_to_str(node_map[0x6000003C0])
