import json
from pathlib import Path


PAIR_FILE = (
    Path(__file__).resolve().parents[1]
    / "mindshift"
    / "data"
    / "pirate_pairs.json"
)


def test_pirate_pairs_are_complete_and_semantically_paired() -> None:
    payload = json.loads(PAIR_FILE.read_text(encoding="utf-8"))

    assert payload["concept"] == "pirate"
    assert len(payload["pairs"]) >= 12
    for pair in payload["pairs"]:
        assert set(pair) == {"prompt", "positive", "negative"}
        assert all(value.strip() for value in pair.values())
        assert pair["positive"] != pair["negative"]
