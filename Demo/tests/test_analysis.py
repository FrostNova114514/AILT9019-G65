from collections import Counter
from types import SimpleNamespace

import pytest

from character_network.analysis import (
    InputValidationError,
    Mention,
    build_graph,
    canonicalize_name,
    count_cooccurrences,
    parse_alias_json,
    validate_text,
)


def test_alias_parsing_and_title_cleanup():
    aliases = parse_alias_json(
        '{"Lizzy": "Elizabeth Bennet", "Miss Bennet": "Elizabeth Bennet"}'
    )

    assert canonicalize_name("Lizzy", aliases) == "Elizabeth Bennet"
    assert canonicalize_name("Miss Bennet", aliases) == "Elizabeth Bennet"
    assert canonicalize_name("Mrs. Bennet", aliases) == "Bennet"
    assert canonicalize_name("Mr. Darcy's", aliases) == "Darcy"


def test_explicit_alias_can_preserve_a_distinguishing_title():
    aliases = parse_alias_json('{"Mrs. Bennet": "Mrs Bennet"}')

    assert canonicalize_name("Mrs. Bennet", aliases) == "Mrs Bennet"


def test_alias_json_must_be_an_object():
    with pytest.raises(InputValidationError, match="JSON object"):
        parse_alias_json('["Lizzy"]')


def test_text_validation_rejects_empty_and_short_input():
    with pytest.raises(InputValidationError):
        validate_text("   ")
    with pytest.raises(InputValidationError, match="too short"):
        validate_text("Elizabeth met Darcy.", require_english=False)


def test_cooccurrence_counts_each_pair_once_per_segment():
    characters, edges, segment_count = count_cooccurrences(
        [
            {"Elizabeth", "Darcy", "Jane"},
            {"Elizabeth", "Darcy"},
            {"Jane"},
        ]
    )

    assert segment_count == 3
    assert characters == Counter({"Elizabeth": 2, "Darcy": 2, "Jane": 2})
    assert edges[("Darcy", "Elizabeth")] == 2
    assert edges[("Darcy", "Jane")] == 1
    assert edges[("Elizabeth", "Jane")] == 1


def test_graph_threshold_and_character_limit():
    mentions = Counter({"Elizabeth": 8, "Darcy": 7, "Jane": 5, "Bingley": 4})
    edges = Counter(
        {
            ("Elizabeth", "Darcy"): 5,
            ("Jane", "Bingley"): 4,
            ("Elizabeth", "Jane"): 1,
        }
    )

    graph = build_graph(
        mentions,
        edges,
        minimum_edge_weight=2,
        maximum_characters=3,
    )

    assert set(graph.nodes) == {"Elizabeth", "Darcy"}
    assert graph["Elizabeth"]["Darcy"]["weight"] == 5


def test_mention_dataclass_is_hashable_and_ordered_data_is_preserved():
    mention = Mention("Elizabeth", 0, 9, 0, 1)
    assert mention.name == "Elizabeth"
    assert SimpleNamespace(value={mention}).value == {mention}
