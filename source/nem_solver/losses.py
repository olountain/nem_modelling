# Pure-Python helpers for working with piecewise-linear (PWL) interconnector loss
# curves. These functions don't touch linopy / the LP at all -- they're useful
# for validating LP outputs against the analytic loss curve and for unit-testing
# the PWL math in isolation.
#
# A PWL loss curve is a sequence of contiguous segments [flow_from, flow_to]
# each carrying a marginal loss factor (MLF). Total losses at flow F are the
# sum over segments of (MLF_l * width_used_l), where width_used_l is how much
# of segment l is consumed by F:
#
#   loss(F) = sum over segments l of MLF_l * min(width_l, max(0, F - start_l))

from nem_solver.schemas import PWLLossSegment


def evaluate_pwl_losses(flow_mw: float, segments: tuple[PWLLossSegment, ...]) -> float:
    """Return total losses at ``flow_mw`` by walking the segment list in order.

    Used for diagnostics and tests -- this is the analytic ground truth that
    the LP's reported losses are compared against.
    """
    if flow_mw < 0.0:
        raise ValueError(f"flow_mw must be non-negative, got {flow_mw}")

    # Walk segments left-to-right, draining `remaining` until it's all consumed.
    # Each segment contributes (MLF * portion_used) to total loss.
    remaining = flow_mw
    losses = 0.0
    for seg in segments:
        width = seg.flow_to_mw - seg.flow_from_mw
        used = min(remaining, width)
        losses += seg.marginal_loss_factor * used
        remaining -= used
        if remaining <= 0.0:
            break

    # If remaining > 0 after the loop, the requested flow exceeds the total
    # coverage of the PWL stack -- the caller asked for an out-of-range flow.
    if remaining > 1e-9:
        raise ValueError(
            f"flow_mw {flow_mw} exceeds total PWL coverage {segments[-1].flow_to_mw}"
        )
    return losses


def fill_segments_in_order(
    flow_mw: float, segments: tuple[PWLLossSegment, ...]
) -> tuple[float, ...]:
    """Return the per-segment MW that the LP optimum would assign at this flow.

    Because MLFs are non-decreasing across segments (convex losses) and we are
    minimising cost, the LP always fills the lowest-MLF segment first, then the
    next, etc. This helper computes that allocation analytically so tests can
    cross-check the LP solution.
    """
    remaining = flow_mw
    out: list[float] = []
    for seg in segments:
        width = seg.flow_to_mw - seg.flow_from_mw
        used = min(max(remaining, 0.0), width)
        out.append(used)
        remaining -= used
    return tuple(out)
