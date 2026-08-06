"""Tests for the sub-agent wiring added on top of the provided triage graph.

`tests/test_graph.py` pins the shape of the graph the exercise ships with and is
deliberately left untouched; everything about the subscriptions and cancellation research
nodes lives here.
"""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

from src.agents import triage as triage_agent
from src.graph import AGENT_MAP, AVAILABLE_AGENTS, build_graph, _make_route_target_fn
from src.main import _log_stream_event
from src.models import MOCK_USER


def _route_call(
    agent_name: str, reason: str = "subscriptions question", call_id: str = "call_1"
):
    """An AIMessage carrying a triage route_to_agent call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "route_to_agent",
                "args": {"agent_name": agent_name, "reason": reason},
                "id": call_id,
            }
        ],
    )


@pytest.fixture
def scripted_llm():
    """A mock LLM whose responses are supplied per test via `.invoke.side_effect`."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.invoke = MagicMock(return_value=AIMessage(content="Hello!"))
    return llm


@pytest.fixture
def graph(scripted_llm):
    return build_graph(scripted_llm)


@pytest.fixture
def node_names(graph):
    return set(graph.get_graph().nodes.keys())


@pytest.fixture
def edges(graph):
    """Every declared edge, static and conditional.

    Read off the builder rather than `get_graph()`: the drawn graph prunes edges
    leaving a node nothing currently routes into, which hides the cancellation research
    agent's wiring until the deterministic handoff is added.
    """
    declared = set(graph.builder.edges)
    for source, branches in graph.builder.branches.items():
        for branch in branches.values():
            declared |= {(source, target) for target in (branch.ends or {}).values()}
    return declared


class TestAgentMap:
    def test_subscriptions_maps_to_its_node(self):
        assert AGENT_MAP["subscriptions"] == "subscriptions_agent"

    def test_cancellation_research_is_not_routable_from_triage(self):
        """FR-25: cancellation research is reachable only via the deterministic handoff."""
        assert "cancellation_research" not in AGENT_MAP

    def test_available_agents_offers_subscriptions(self):
        assert "subscriptions" in AVAILABLE_AGENTS

    def test_available_agents_omits_cancellation_research(self):
        assert "cancellation_research" not in AVAILABLE_AGENTS

    def test_every_available_agent_has_a_node_in_the_map(self):
        assert set(AVAILABLE_AGENTS) <= set(AGENT_MAP)

    def test_available_agents_descriptions_are_non_empty(self):
        assert all(description.strip() for description in AVAILABLE_AGENTS.values())

    def test_subscriptions_description_mentions_subscriptions(self):
        assert "subscription" in AVAILABLE_AGENTS["subscriptions"].lower()

    def test_routing_function_resolves_subscriptions(self):
        route = _make_route_target_fn(AGENT_MAP)
        state = {
            "messages": [_route_call("subscriptions")],
            "current_agent": "triage",
            "user_info": {},
        }
        assert route(state) == "subscriptions_agent"

    def test_routing_function_cannot_reach_cancellation_research(self):
        route = _make_route_target_fn(AGENT_MAP)
        state = {
            "messages": [_route_call("cancellation_research")],
            "current_agent": "triage",
            "user_info": {},
        }
        assert route(state) == "unavailable_agent"


class TestTriagePrompt:
    def test_prompt_lists_subscriptions(self):
        prompt = triage_agent._build_system_prompt("Sarah", AVAILABLE_AGENTS)
        assert '"subscriptions"' in prompt

    def test_prompt_does_not_list_cancellation_research(self):
        prompt = triage_agent._build_system_prompt("Sarah", AVAILABLE_AGENTS)
        assert "cancellation_research" not in prompt.lower()

    def test_prompt_is_not_the_no_agents_variant(self):
        prompt = triage_agent._build_system_prompt("Sarah", AVAILABLE_AGENTS)
        assert "no specialist agents are available" not in prompt


class TestGraphNodes:
    @pytest.mark.parametrize(
        "name",
        [
            "subscriptions_agent",
            "subscriptions_tools",
            "cancellation_research_agent",
            "cancellation_research_tools",
        ],
    )
    def test_node_present(self, node_names, name):
        assert name in node_names

    def test_provided_nodes_still_present(self, node_names):
        assert {"triage", "triage_tools", "unavailable_agent"} <= node_names


class TestGraphEdges:
    def test_subscriptions_tools_returns_to_its_agent(self, edges):
        assert ("subscriptions_tools", "subscriptions_agent") in edges

    def test_cancellation_research_tools_returns_to_its_agent(self, edges):
        assert ("cancellation_research_tools", "cancellation_research_agent") in edges

    def test_triage_can_reach_subscriptions_agent(self, edges):
        assert ("triage", "subscriptions_agent") in edges

    def test_subscriptions_agent_can_reach_its_tool_node(self, edges):
        assert ("subscriptions_agent", "subscriptions_tools") in edges

    def test_subscriptions_agent_returns_to_triage(self, edges):
        assert ("subscriptions_agent", "triage") in edges

    def test_cancellation_research_agent_can_reach_its_tool_node(self, edges):
        assert ("cancellation_research_agent", "cancellation_research_tools") in edges


class TestGoldenPath:
    """A routing request reaches the subscriptions agent and comes back to triage."""

    @pytest.fixture
    def visits(self, scripted_llm):
        scripted_llm.invoke.side_effect = [
            _route_call("subscriptions", "wants to review recurring payments"),
            AIMessage(content="You are committed to £1,507.39 a year."),
            AIMessage(content="Anything else I can help with?"),
        ]
        graph = build_graph(scripted_llm)

        seen = []
        for update in graph.stream(
            {
                "messages": [HumanMessage(content="What am I paying for each month?")],
                "user_info": MOCK_USER,
                "current_agent": "triage",
            },
            stream_mode="updates",
        ):
            seen.extend(update.keys())
        return seen

    def test_reaches_the_subscriptions_agent(self, visits):
        assert "subscriptions_agent" in visits

    def test_does_not_fall_through_to_unavailable(self, visits):
        assert "unavailable_agent" not in visits

    def test_returns_to_triage_after_the_sub_agent(self, visits):
        assert visits.index("triage") < visits.index("subscriptions_agent")
        assert "triage" in visits[visits.index("subscriptions_agent") :]

    def test_cancellation_research_is_not_entered_on_the_golden_path(self, visits):
        assert "cancellation_research_agent" not in visits

    def test_final_state_carries_the_sub_agent_answer(self, scripted_llm):
        scripted_llm.invoke.side_effect = [
            _route_call("subscriptions"),
            AIMessage(content="You are committed to £1,507.39 a year."),
            AIMessage(content="Anything else?"),
        ]
        graph = build_graph(scripted_llm)
        final = graph.invoke(
            {
                "messages": [HumanMessage(content="What am I paying for?")],
                "user_info": MOCK_USER,
                "current_agent": "triage",
            }
        )
        contents = [m.content for m in final["messages"]]
        assert "You are committed to £1,507.39 a year." in contents

    def test_dangling_route_call_is_answered(self, scripted_llm):
        """The unexecuted route_to_agent call must not reach the provider unanswered."""
        scripted_llm.invoke.side_effect = [
            _route_call("subscriptions", call_id="call_dangling"),
            AIMessage(content="Here is the breakdown."),
            AIMessage(content="Anything else?"),
        ]
        graph = build_graph(scripted_llm)
        final = graph.invoke(
            {
                "messages": [HumanMessage(content="What am I paying for?")],
                "user_info": MOCK_USER,
                "current_agent": "triage",
            }
        )
        answered = {getattr(m, "tool_call_id", None) for m in final["messages"]}
        assert "call_dangling" in answered


class TestCliTranscript:
    """FR-19 is only demonstrably wired if the routing line shows up in the CLI."""

    def _transcript(self, agent_name: str) -> str:
        console = Console(record=True, width=200)
        _log_stream_event(
            console,
            "triage",
            {
                "messages": [
                    _route_call(agent_name, "wants to review recurring payments")
                ]
            },
        )
        return console.export_text()

    def test_routing_line_names_subscriptions(self):
        assert "> Routing: → subscriptions" in self._transcript("subscriptions")

    def test_routing_line_carries_the_reason(self):
        assert "wants to review recurring payments" in self._transcript("subscriptions")
