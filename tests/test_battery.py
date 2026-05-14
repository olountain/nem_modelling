import pytest
from nem_solver import solve_dispatch


def test_battery_discharges_at_high_price(battery_high_price_snapshot):
    """Demand 350, cheap supplies 300, peaker priced $300 with 200 MW. Battery discharge bid $200,
    charge bid $20. Battery should discharge fully (50 MW) since RRP > $200, displacing 50 MW
    of $300 peaker."""
    result = solve_dispatch(battery_high_price_snapshot, resolution_minutes=5)
    [batt] = result.storage_dispatch
    assert batt.discharge_mw == pytest.approx(50.0)
    assert batt.charge_mw == pytest.approx(0.0)
    # peaker is on the margin -> RRP = $300
    assert result.rrp_by_region["NSW1"] == pytest.approx(300.0)


def test_battery_charges_at_low_price(battery_low_price_snapshot):
    """Demand 100 MW, only $10 generator. Battery charge WTP $50 -> charges (50 MW).
    The generator now serves 100 + 50 = 150 MW."""
    result = solve_dispatch(battery_low_price_snapshot, resolution_minutes=5)
    [batt] = result.storage_dispatch
    assert batt.charge_mw == pytest.approx(50.0)
    assert batt.discharge_mw == pytest.approx(0.0)
    # marginal generator price = $10
    assert result.rrp_by_region["NSW1"] == pytest.approx(10.0)
    [gen] = result.unit_dispatch
    assert gen.dispatched_mw == pytest.approx(150.0)


def test_no_simultaneous_charge_discharge(battery_high_price_snapshot):
    """Lock in the assumption that the LP optimum never has both charge and discharge."""
    result = solve_dispatch(battery_high_price_snapshot, resolution_minutes=5)
    [batt] = result.storage_dispatch
    assert batt.discharge_mw * batt.charge_mw == pytest.approx(0.0, abs=1e-6)
