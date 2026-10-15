"""Checkpoint ranking and the global champion file."""

from lander.atomic import read_json
from lander.champion import CHAMPION_FILE, score, try_update_champion


def rec(name, mean, success=None):
    return {"run_name": name, "model_path": f"{name}.zip", "mean_reward": mean, "success_rate": success}


def test_success_rate_outranks_return():
    assert score(rec("a", 100.0, 0.9)) > score(rec("b", 300.0, 0.8))
    assert score(rec("a", 200.0, 0.9)) > score(rec("b", 100.0, 0.9))  # tie on success: return decides


def test_without_success_info_ranks_by_return():
    assert score(rec("a", 200.0)) > score(rec("b", 100.0))


def test_champion_only_replaced_by_a_better_score(tmp_path):
    assert try_update_champion(tmp_path, rec("a", 250.0, 0.80))[0]
    assert not try_update_champion(tmp_path, rec("b", 300.0, 0.75))[0]  # more return, less success
    assert try_update_champion(tmp_path, rec("c", 200.0, 0.95))[0]
    assert read_json(tmp_path / CHAMPION_FILE)["run_name"] == "c"
