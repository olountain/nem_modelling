from datetime import datetime, timedelta

import pytest
from nem_solver import (
    BidBand,
    DispatchSnapshot,
    Generator,
    Region,
    run_timeseries,
)


def _snapshot(ts: datetime, demand_mw: float) -> DispatchSnapshot:
    return DispatchSnapshot(
        timestamp=ts,
        regions=(Region(region_id="NSW1", demand_mw=demand_mw),),
        generators=(
            Generator(
                unit_id="G1",
                region_id="NSW1",
                bands=(
                    BidBand(price=30.0, quantity=50.0),
                    BidBand(price=80.0, quantity=50.0),
                    BidBand(price=200.0, quantity=100.0),
                ),
                max_capacity_mw=200.0,
            ),
        ),
        interconnectors=(),
    )


def test_timeseries_smoke_5_minute():
    base = datetime(2026, 5, 8, 0, 0)
    demands = [40.0, 60.0, 90.0, 120.0, 180.0]
    snapshots = [_snapshot(base + timedelta(minutes=5 * i), d) for i, d in enumerate(demands)]
    ts = run_timeseries(snapshots, resolution_minutes=5)

    assert list(ts.prices.columns) == ["NSW1"]
    assert len(ts.prices) == len(demands)
    # band 1 covers 0-50 MW at $30; band 2 50-100 MW at $80; band 3 100-200 MW at $200.
    # demands [40, 60, 90, 120, 180] -> marginal bands [1, 2, 2, 3, 3]
    expected_rrps = [30.0, 80.0, 80.0, 200.0, 200.0]
    for ts_row, expected in zip(ts.prices["NSW1"].tolist(), expected_rrps, strict=True):
        assert ts_row == pytest.approx(expected)

    # dispatch DataFrame is long-form
    assert set(ts.dispatch.columns) == {"timestamp", "unit_id", "dispatched_mw"}
    assert len(ts.dispatch) == len(demands)


def test_timeseries_smoke_30_minute():
    base = datetime(2026, 5, 8, 0, 0)
    # Avoid demand at exact band boundaries (50, 100) where the LP dual is degenerate.
    demands = [60.0, 90.0]
    snapshots = [_snapshot(base + timedelta(minutes=30 * i), d) for i, d in enumerate(demands)]
    ts = run_timeseries(snapshots, resolution_minutes=30)
    # demand 60 MW uses band 1 fully + 10 MW of band 2 -> RRP = $80
    # demand 90 MW uses band 1 + 40 MW of band 2 -> RRP = $80
    assert ts.prices["NSW1"].iloc[0] == pytest.approx(80.0)
    assert ts.prices["NSW1"].iloc[1] == pytest.approx(80.0)
