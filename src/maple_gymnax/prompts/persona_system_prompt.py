"""Persona System Prompt Formulator for LLM WZ-to-Tensor Translation."""

from typing import Optional


class PersonaSystemPrompt:
    """Master persona prompt instructing LLMs to transform WZ data to physical tensors."""

    MASTER_PROMPT: str = (
        "You are the MapleStory Gymnax Physics Engineer persona. Your objective is to extract "
        "raw KMS WZ client nodes (BossSuu.img.json, bossSuu.img.json), anchor coordinates, frame delays, "
        "and physical collision volumes, and transform them into branch-free JAX/XLA tensor simulation "
        "primitives conforming to gymnax.environments.environment.Environment. Adhere strictly to Flax "
        "struct dataclasses, static array dimensions, and vector arithmetic without Python conditionals."
    )

    @classmethod
    def get_system_prompt(cls) -> str:
        """Returns the base system instruction persona."""
        return cls.MASTER_PROMPT

    @classmethod
    def format_extraction_prompt(
        cls,
        wz_node_name: str,
        raw_json_snippet: str,
        target_schema: Optional[str] = "Draft 2020-12",
    ) -> str:
        """Formats a concrete LLM extraction prompt for a given WZ node snippet."""
        return (
            f"{cls.MASTER_PROMPT}\n\n"
            f"Target Node: {wz_node_name}\n"
            f"Target Schema: {target_schema}\n"
            f"Raw Data:\n{raw_json_snippet}\n\n"
            "Task: Output validated EnvParams properties and branch-free step_env transitions."
        )
