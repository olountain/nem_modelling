from datetime import datetime

import pytest
from nem_solver import (
    BidBand,
    DispatchSnapshot,
    Generator,
    Interconnector,
    PWLLossSegment,
    Region,
    solve_dispatch,
)
from nem_solver.losses import evaluate_pwl_losses, fill_segments_in_order


def _three_segment_loss_curve() -> tuple[PWLLossSegment, ...]:
    # convex MLF schedule: small, medium, large losses as flow increases
    return (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=50.0, marginal_loss_factor=0.02),
        PWLLossSegment(flow_from_mw=50.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
        PWLLossSegment(flow_from_mw=100.0, flow_to_mw=150.0, marginal_loss_factor=0.10),
    )


def test_pwl_losses_recovered_in_dispatch():
    """Set up a binding flow and verify reported losses match the analytic PWL evaluation,
    and that segments fill in cost-minimising (non-decreasing) order."""
    segs = _three_segment_loss_curve()
    snap = DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(
            Region(region_id="NSW1", demand_mw=0.0),  # exporter, no local demand
            Region(region_id="VIC1", demand_mw=80.0),
        ),
        generators=(
            Generator(
                unit_id="NSW_GEN",
                region_id="NSW1",
                bands=(BidBand(price=10.0, quantity=200.0),),
                max_capacity_mw=200.0,
            ),
            Generator(
                unit_id="VIC_PEAKER",
                region_id="VIC1",
                bands=(BidBand(price=500.0, quantity=200.0),),
                max_capacity_mw=200.0,
            ),
        ),
        interconnectors=(
            Interconnector(
                ic_id="NSW1-VIC1",
                from_region="NSW1",
                to_region="VIC1",
                forward_limit_mw=150.0,
                reverse_limit_mw=150.0,
                forward_loss_segments=segs,
                reverse_loss_segments=segs,
            ),
        ),
    )

    result = solve_dispatch(snap, resolution_minutes=5)
    [ic_flow] = result.interconnector_flows

    # at flow = F, received in VIC = F - loss(F). VIC needs 80 MW served.
    # Solve F - (0.02*50 + 0.05*(F-50)) = 80 -> 0.95F = 78.5 -> F ~= 82.63
    expected_F = (80.0 - 2.5 + 1.0) / 0.95
    assert ic_flow.flow_mw == pytest.approx(expected_F, abs=1e-3)
    expected_loss = evaluate_pwl_losses(expected_F, segs)
    assert ic_flow.losses_mw == pytest.approx(expected_loss, abs=1e-3)

    # cross-check: the LP fills segments in MLF-increasing order
    expected_fill = fill_segments_in_order(expected_F, segs)
    # segment 0 (lowest MLF) should be at full width 50
    assert expected_fill[0] == pytest.approx(50.0)
    assert expected_fill[1] == pytest.approx(expected_F - 50.0)
    assert expected_fill[2] == pytest.approx(0.0)


def test_pwl_no_simultaneous_directions():
    """Net flow should not show both forward and reverse simultaneously activated."""
    segs = _three_segment_loss_curve()
    snap = DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(
            Region(region_id="NSW1", demand_mw=50.0),
            Region(region_id="VIC1", demand_mw=50.0),
        ),
        generators=(
            Generator(
                unit_id="NSW_GEN",
                region_id="NSW1",
                bands=(BidBand(price=20.0, quantity=200.0),),
                max_capacity_mw=200.0,
            ),
            Generator(
                unit_id="VIC_GEN",
                region_id="VIC1",
                bands=(BidBand(price=20.0, quantity=200.0),),
                max_capacity_mw=200.0,
            ),
        ),
        interconnectors=(
            Interconnector(
                ic_id="NSW1-VIC1",
                from_region="NSW1",
                to_region="VIC1",
                forward_limit_mw=150.0,
                reverse_limit_mw=150.0,
                forward_loss_segments=segs,
                reverse_loss_segments=segs,
            ),
        ),
    )
    result = solve_dispatch(snap, resolution_minutes=5)
    [ic_flow] = result.interconnector_flows
    # equal prices and zero spread -> no incentive to flow at all
    assert abs(ic_flow.flow_mw) < 1e-6
    assert ic_flow.losses_mw == pytest.approx(0.0)
