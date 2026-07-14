from crypto_event_trader.research import return_statistics


def test_return_statistics_are_deterministic() -> None:
    result = return_statistics([0.01, 0.02, -0.005, 0.03], bootstrap_samples=200)
    assert result.observations == 4
    assert result.hit_rate == 0.75
    assert result.confidence_interval_95[0] <= result.mean <= result.confidence_interval_95[1]
