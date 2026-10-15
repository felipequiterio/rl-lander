"""Non-learned controllers, usable anywhere a trained policy is (same predict(obs) API).

CONTROLLERS maps a name to a factory taking the (unwrapped) env, so a controller
can read the env's parameters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lander.controllers.zem import ZemZevController

CONTROLLERS: dict[str, Callable[[Any], Any]] = {
    "zem": lambda env: ZemZevController(env.unwrapped.cfg),
}
