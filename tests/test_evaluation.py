from mindshift.evaluation import non_ascii_ratio, repetition_ratio
from scripts.evaluate import render_report


def test_repetition_ratio_is_case_insensitive() -> None:
    assert repetition_ratio("Ship ship sails home") == 0.25
    assert repetition_ratio("") == 0.0


def test_non_ascii_ratio() -> None:
    assert non_ascii_ratio("abc") == 0.0
    assert non_ascii_ratio("aé") == 0.5
    assert non_ascii_ratio("") == 0.0


def test_report_marks_alpha_zero_equality() -> None:
    report = render_report(
        [
            {
                "prompt": "Test prompt",
                "alpha": 0.0,
                "baseline": "Same",
                "steered": "Same",
                "repetition_ratio": 0.0,
                "non_ascii_ratio": 0.0,
                "latency_ms": 10.0,
            }
        ],
        backend_url="http://localhost:8000",
        concept="pirate",
        layer=12,
        seed=42,
    )

    assert "Alpha=0 exact equality: **PASS**" in report
    assert "### Alpha 0: Test prompt" in report
