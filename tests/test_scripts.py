"""Guards on the shell drivers, which the unit tests cannot exercise directly."""

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

GENERATION_STAGES = ["01_extract.sh", "03_steer.sh", "04_controls.sh", "04b_ablate.sh"]


@pytest.mark.parametrize("name", GENERATION_STAGES)
def test_generation_stages_use_the_configured_prompt_set(name):
    """A stage that silently falls back to the CLI default evaluates other prompts."""
    body = (SCRIPTS / name).read_text(encoding="utf-8")
    assert '--prompt-set "$PROMPT_SET"' in body


@pytest.mark.parametrize("name", GENERATION_STAGES)
def test_generation_stages_default_to_all_prompts(name):
    body = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "PROMPT_SET=${PROMPT_SET:-all}" in body


def test_controls_do_not_inherit_the_full_sweep_grid():
    body = (SCRIPTS / "04_controls.sh").read_text(encoding="utf-8")
    assert "CONTROL_COEFS" in body
    assert "--coefs $CONTROL_COEFS" in body


def test_block3_sweeps_layers_and_uses_the_centroid_control():
    script = (SCRIPTS / "14_headroom_and_centroid.sh").read_text(encoding="utf-8")
    assert "--layers $layers" in script
    assert "for control in centroid random" in script
    assert "--temperature" in script
