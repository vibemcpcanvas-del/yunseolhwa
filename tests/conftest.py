"""Pytest configuration and shared fixtures for maple_gymnax test suites."""

from __future__ import annotations

import os
from typing import Any, Dict
import pytest

from maple_gymnax.parser.schema import EnvParams


@pytest.fixture
def default_env_params() -> EnvParams:
    """Fixture returning canonical default EnvParams instance."""
    return EnvParams()


@pytest.fixture
def mp_dir() -> str:
    """Fixture returning path to C:\\mp directory."""
    return r"C:\mp"


@pytest.fixture
def sample_raw_wz_node() -> Dict[str, Any]:
    """Fixture returning a raw WZ Extractor node structure for parser testing."""
    return {
        "name": "1001",
        "type": "SubProperty",
        "children": [
            {
                "name": "000",
                "type": "SubProperty",
                "children": [
                    {
                        "name": "0",
                        "type": "Canvas",
                        "width": 120,
                        "height": 200,
                        "children": [
                            {"name": "delay", "type": "Int", "value": 120},
                            {"name": "origin", "type": "Vector", "value": [60, 100]},
                        ],
                    },
                    {
                        "name": "uol_ref",
                        "type": "UOL",
                        "target": "../0",
                    },
                ],
            }
        ],
    }


@pytest.fixture
def sample_spine_atlas_text() -> str:
    """Fixture returning a sample Spine texture atlas text block."""
    return """
Swoo_Bossmap_Phase1.png
size: 2046, 2037
format: RGBA8888
filter: Linear, Linear
repeat: none
02_upper ligh10_1
  rotate: false
  xy: 80, 2005
  size: 35, 30
  orig: 35, 30
  offset: 0, 0
  index: -1
03_blue_light01_1
  rotate: true
  xy: 906, 1799
  size: 107, 130
  orig: 107, 130
  offset: 0, 0
  index: -1
"""
