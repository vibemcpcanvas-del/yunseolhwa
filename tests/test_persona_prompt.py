"""Unit tests for Persona System Prompt Formulator."""

import pytest
from maple_gymnax.prompts.persona_system_prompt import PersonaSystemPrompt


class TestPersonaSystemPrompt:
    """Validates persona instructions and extraction prompt formatting."""

    def test_system_prompt_contains_key_directives(self):
        prompt = PersonaSystemPrompt.get_system_prompt()
        assert "MapleStory Gymnax Physics Engineer persona" in prompt
        assert "BossSuu.img.json" in prompt
        assert "branch-free JAX/XLA tensor simulation" in prompt
        assert "gymnax.environments.environment.Environment" in prompt
        assert "Flax" in prompt

    def test_format_extraction_prompt(self):
        snippet = '{"attack1": {"delay": 120, "rect": [-50, -50, 50, 50]}}'
        formatted = PersonaSystemPrompt.format_extraction_prompt(
            wz_node_name="Mob/8950000.img/attack1",
            raw_json_snippet=snippet,
        )
        assert "Target Node: Mob/8950000.img/attack1" in formatted
        assert snippet in formatted
        assert "Draft 2020-12" in formatted
