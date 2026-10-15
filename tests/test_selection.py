import pytest

from embodied_research.selection import choose_candidate


def groups(baseline=.5, low=.5, medium=.5, high=.375):
    return [{"method": name, "macro_task_success": success, "mean_inference_calls": calls,
             "model": "minerva", "condition": "clean"}
            for name, success, calls in [("fixed8", baseline, 40), ("adaptive_low", low, 90),
                                        ("adaptive_medium", medium, 35), ("adaptive_high", high, 22)]]


def test_selection_requires_baseline_success_before_minimizing_calls():
    result = choose_candidate(groups())
    assert result["selected"] == "adaptive_medium"
    assert "adaptive_high" not in result["eligible"]
    assert result["fallback_used"] is False


def test_no_qualifier_uses_best_success_instead_of_cheapest_failure():
    result = choose_candidate(groups(baseline=.75, low=.5, medium=.375, high=.25))
    assert result["selected"] == "adaptive_low"
    assert result["fallback_used"] is True


def test_selection_cannot_silently_drop_a_candidate_or_mix_conditions():
    with pytest.raises(ValueError, match="exactly"):
        choose_candidate(groups()[:-1])
    sample = groups()
    sample[0]["condition"] = "delay4"
    with pytest.raises(ValueError, match="development data"):
        choose_candidate(sample)
