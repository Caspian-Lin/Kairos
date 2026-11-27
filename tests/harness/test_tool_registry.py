import pytest

from kairos.config import BotProfile
from kairos.harness.tool_registry import ToolRegistry
from kairos.models import ToolDefinition, ToolSpec


def _spec(
    name: str, profiles: tuple[str, ...] = ("utility_assistant", "social_companion")
) -> ToolSpec:
    return ToolSpec(
        definition=ToolDefinition(
            name=name, description="Full description", parameters={"type": "object"}
        ),
        catalog_hint="Get a short result",
        allowed_profiles=profiles,
    )


def test_registry_filters_by_profile_and_selected_tools() -> None:
    status = _spec("status")
    reserved = _spec("reserved", ("utility_assistant",))
    notes = _spec("notes")
    registry = ToolRegistry([status, reserved, notes])
    profile = BotProfile(bot_id="1", self_id=1, nickname="Test", persona_prompt="Test")

    assert registry.catalog_prompt(profile) == ""
    assert registry.resolve_definitions(profile, ["status"]) == []

    profile.enabled_tools = ["unknown", "status", "reserved", "status"]
    assert registry.catalog_prompt(profile) == "status: Get a short result"
    assert registry.resolve_definitions(profile, ["reserved", "status", "status", "notes"]) == [
        status.definition
    ]
    profile.profile_type = "utility_assistant"
    assert registry.catalog_prompt(profile) == (
        "status: Get a short result | reserved: Get a short result"
    )
    assert registry.resolve_definitions(profile, ["reserved", "status"]) == [
        reserved.definition,
        status.definition,
    ]


def test_register_rejects_duplicate_definition_names() -> None:
    registry = ToolRegistry()
    registry.register(_spec("notes"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_spec("notes"))
