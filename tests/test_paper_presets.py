from __future__ import annotations

from tvbtoolkit.workflows.presets import maria_sacha_nature_conditions


def test_maria_sacha_nature_conditions_names() -> None:
    conds = maria_sacha_nature_conditions()
    names = [c.name for c in conds]
    assert names == ["wake", "nmda", "gaba", "sleep"]


def test_maria_sacha_nature_conditions_have_overrides() -> None:
    conds = maria_sacha_nature_conditions()
    for c in conds:
        assert isinstance(c.parameter_overrides, dict)
        assert "parameter_model" in c.parameter_overrides
