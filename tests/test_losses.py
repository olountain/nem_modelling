import pytest
from nem_solver import PWLLossSegment
from nem_solver.losses import evaluate_pwl_losses, fill_segments_in_order


def test_zero_flow_zero_losses():
    segs = (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
    )
    assert evaluate_pwl_losses(0.0, segs) == 0.0


def test_single_segment_loss_is_mlf_times_flow():
    segs = (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
    )
    assert evaluate_pwl_losses(40.0, segs) == pytest.approx(2.0)


def test_multi_segment_fills_in_order():
    segs = (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=50.0, marginal_loss_factor=0.02),
        PWLLossSegment(flow_from_mw=50.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
        PWLLossSegment(flow_from_mw=100.0, flow_to_mw=150.0, marginal_loss_factor=0.10),
    )
    # flow 80 — fills seg0 fully (50), seg1 partial (30), seg2 nothing
    assert fill_segments_in_order(80.0, segs) == pytest.approx((50.0, 30.0, 0.0))
    # corresponding loss: 0.02*50 + 0.05*30 + 0 = 1 + 1.5 = 2.5
    assert evaluate_pwl_losses(80.0, segs) == pytest.approx(2.5)


def test_evaluate_rejects_negative_flow():
    segs = (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
    )
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_pwl_losses(-1.0, segs)


def test_evaluate_rejects_overflow():
    segs = (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
    )
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_pwl_losses(150.0, segs)
