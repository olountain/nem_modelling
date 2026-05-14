import pytest
from nem_solver import solve_dispatch


def test_two_region_unconstrained_interconnector(two_region_snapshot):
    result = solve_dispatch(two_region_snapshot, resolution_minutes=5)
    assert result.solver_status == "ok"
    # NSW should fully serve VIC's demand because it's cheaper
    rrp = result.rrp_by_region
    assert rrp["NSW1"] == pytest.approx(40.0)
    assert rrp["VIC1"] == pytest.approx(40.0)  # IC unconstrained, prices equalise
    [ic_flow] = result.interconnector_flows
    assert ic_flow.flow_mw == pytest.approx(100.0)  # NSW -> VIC, equal to VIC's demand
    assert ic_flow.losses_mw == pytest.approx(0.0)
    nsw_gen = next(u for u in result.unit_dispatch if u.unit_id == "NSW_GEN")
    vic_gen = next(u for u in result.unit_dispatch if u.unit_id == "VIC_GEN")
    assert nsw_gen.dispatched_mw == pytest.approx(200.0)  # NSW demand 100 + 100 export
    assert vic_gen.dispatched_mw == pytest.approx(0.0)


def test_two_region_binding_interconnector(two_region_constrained_snapshot):
    result = solve_dispatch(two_region_constrained_snapshot, resolution_minutes=5)
    rrp = result.rrp_by_region
    assert rrp["NSW1"] == pytest.approx(40.0)
    assert rrp["VIC1"] == pytest.approx(120.0)  # VIC peaker is on the margin
    [ic_flow] = result.interconnector_flows
    assert ic_flow.flow_mw == pytest.approx(50.0)  # IC binding at limit
    nsw_gen = next(u for u in result.unit_dispatch if u.unit_id == "NSW_GEN")
    vic_gen = next(u for u in result.unit_dispatch if u.unit_id == "VIC_GEN")
    assert nsw_gen.dispatched_mw == pytest.approx(150.0)  # 100 local + 50 export
    assert vic_gen.dispatched_mw == pytest.approx(50.0)  # rest from VIC peaker
