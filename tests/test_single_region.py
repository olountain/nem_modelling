import pytest
from nem_solver import solve_dispatch


def test_single_region_single_generator(single_region_snapshot):
    result = solve_dispatch(single_region_snapshot, resolution_minutes=5)
    assert result.solver_status == "ok"
    assert result.rrp_by_region["NSW1"] == pytest.approx(50.0)
    [g] = result.unit_dispatch
    assert g.unit_id == "G1"
    assert g.dispatched_mw == pytest.approx(60.0)
    # 60 MW * $50/MWh * (5/60) hours = $250
    assert result.objective_value == pytest.approx(250.0)


def test_band_stacking(stacked_bands_snapshot):
    result = solve_dispatch(stacked_bands_snapshot, resolution_minutes=5)
    [g] = result.unit_dispatch
    assert g.dispatched_mw == pytest.approx(50.0)
    assert g.band_dispatch_mw == pytest.approx((20.0, 30.0, 0.0))
    # marginal band sets RRP
    assert result.rrp_by_region["NSW1"] == pytest.approx(60.0)


def test_rrp_equals_marginal_band_price(stacked_bands_snapshot):
    result = solve_dispatch(stacked_bands_snapshot, resolution_minutes=30)
    # marginal band is $60 regardless of resolution
    assert result.rrp_by_region["NSW1"] == pytest.approx(60.0)
