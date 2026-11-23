from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .bandits import choose_epsilon_greedy, choose_lingape, choose_thompson, choose_ucb
from .base import AcquisitionContext
from .optimal_design import choose_a_opt, choose_d_opt, choose_e_opt, choose_g_opt
from .proposed import choose_hard, choose_soft

Chooser = Callable[[AcquisitionContext], int]


CHOOSERS: dict[str, Chooser] = {
    "hard": choose_hard,
    "soft": choose_soft,
    "a_opt": choose_a_opt,
    "d_opt": choose_d_opt,
    "e_opt": choose_e_opt,
    "g_opt": choose_g_opt,
    "ucb": choose_ucb,
    "thompson": choose_thompson,
    "epsilon_greedy": choose_epsilon_greedy,
    "lingape": choose_lingape,
}


def choose_one(method_name: str, context: AcquisitionContext) -> int:
    if method_name == "random":
        return int(context.rng.integers(0, context.X.shape[0]))
    if method_name not in CHOOSERS:
        raise ValueError(f"Unknown method: {method_name}. Available: {sorted(CHOOSERS)}")
    return CHOOSERS[method_name](context)


__all__ = ["AcquisitionContext", "choose_one"]
