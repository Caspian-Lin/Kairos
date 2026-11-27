"""Filter configured tools before presenting them to the gate or provider."""

from collections.abc import Iterable, Sequence

from kairos.config import BotProfile
from kairos.models import ToolDefinition, ToolSpec
from kairos.protocols import ToolRegistryProtocol


class ToolRegistry(ToolRegistryProtocol):
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        name = spec.definition.name
        if name in self._specs:
            raise ValueError(f"Tool {name!r} is already registered")
        self._specs[name] = spec

    def _allowed_specs(self, profile: BotProfile) -> Iterable[ToolSpec]:
        for name in dict.fromkeys(profile.enabled_tools):
            if (spec := self._specs.get(name)) and profile.profile_type in spec.allowed_profiles:
                yield spec

    def catalog_prompt(self, profile: BotProfile) -> str:
        return " | ".join(
            f"{spec.definition.name}: {' '.join(spec.catalog_hint.split())}"
            for spec in self._allowed_specs(profile)
        )

    def resolve_definitions(
        self, profile: BotProfile, names: Sequence[str]
    ) -> list[ToolDefinition]:
        if not names or not profile.enabled_tools:
            return []
        allowed = {spec.definition.name: spec.definition for spec in self._allowed_specs(profile)}
        return [allowed[name] for name in dict.fromkeys(names) if name in allowed]
