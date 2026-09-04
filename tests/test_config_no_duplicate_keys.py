"""Config YAMLs must not silently drop settings to duplicate keys.

The now-removed `configs/sand.yaml` carried TWO top-level `simulation:` blocks.
YAML takes the last one, so the entire first block -- dt, substeps,
settle_steps, every settle threshold and all their justifying comments -- was
discarded, and the only survivor was the two-line block appended later. Nothing
warned, and the runs used `settle_steps = 100` where the config said 2500, so
the pile was recorded still moving (docs/rejected_mpm_sand.md).

The sand path is gone; this check is not, because the failure mode belongs to
YAML rather than to sand and every remaining config is just as exposed.

This is the same failure shape as `safety_margin` (declared 0.005, hardcoded
0.02, never read) and the fixed `rigid_options` allow-list: a config that lies
about what the simulation is doing. `yaml.safe_load` cannot catch it, so the
check has to be explicit.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "Genesis" / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader that raises instead of letting a later key win."""


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1} "
                f"(first seen line {mapping[key] + 1}); YAML keeps only the "
                f"last, so the earlier block is silently discarded")
        mapping[key] = key_node.start_mark.line
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_has_no_duplicate_keys(path):
    yaml.load(path.read_text(), Loader=_DuplicateKeyLoader)
