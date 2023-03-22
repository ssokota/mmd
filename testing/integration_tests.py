"""Tests to check whether examples run

Tests do not attempt to check correctness
"""

from importlib import util
import os
import subprocess


def test_perturbed_rps() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    script = f"{here}/../examples/nfg/perturbed_rps.py"
    spec = util.spec_from_file_location("perturbed_rps", script)
    assert spec is not None and spec.loader is not None
    perturbed_rps = util.module_from_spec(spec)
    spec.loader.exec_module(perturbed_rps)
    for temperature in perturbed_rps.temperature_choices:
        subprocess.check_call(
            ["python", script, "--temperature", f"{temperature}", "--dry_run"]
        )


def test_nash() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    script = f"{here}/../examples/efg/nash.py"
    spec = util.spec_from_file_location("nash", script)
    assert spec is not None and spec.loader is not None
    nash = util.module_from_spec(spec)
    spec.loader.exec_module(nash)
    for game in nash.game_choices:
        for approach in nash.approach_choices:
            subprocess.check_call(
                [
                    "python",
                    script,
                    "--game",
                    f"{game}",
                    "--approach",
                    f"{approach}",
                    "--dry_run",
                ]
            )


def test_cfr() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    script = f"{here}/../examples/efg/cfr.py"
    spec = util.spec_from_file_location("cfr", script)
    assert spec is not None and spec.loader is not None
    cfr = util.module_from_spec(spec)
    spec.loader.exec_module(cfr)
    for game in cfr.game_choices:
        for variant in cfr.variant_choices:
            subprocess.run(
                [
                    "python",
                    script,
                    "--game",
                    f"{game}",
                    "--variant",
                    f"{variant}",
                    "--dry_run",
                ]
            )


def test_aqre() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    script = f"{here}/../examples/efg/aqre.py"
    spec = util.spec_from_file_location("aqre", script)
    assert spec is not None and spec.loader is not None
    aqre = util.module_from_spec(spec)
    spec.loader.exec_module(aqre)
    for game in aqre.game_choices:
        for temperature in aqre.temperature_choices:
            subprocess.check_call(
                [
                    "python",
                    script,
                    "--game",
                    f"{game}",
                    "--temperature",
                    f"{temperature}",
                    "--dry_run",
                ]
            )


def test_evaluate() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    script = f"{here}/../examples/h2h/evaluate.py"
    spec = util.spec_from_file_location("evaluate", script)
    assert spec is not None and spec.loader is not None
    evaluate = util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    for game in evaluate.game_choices:
        for agent1 in evaluate.agent_choices:
            for agent2 in evaluate.agent_choices:
                subprocess.check_call(
                    [
                        "python",
                        script,
                        "--game",
                        f"{game}",
                        "--agent1",
                        f"{agent1}",
                        "--agent2",
                        f"{agent2}",
                        "--dry_run",
                    ],
                    stdout=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    test_perturbed_rps()
    test_nash()
    test_cfr()
    test_aqre()
    test_evaluate()
