# Multi-period orchestration: run solve_dispatch() over a sequence of snapshots
# and collect the results into tidy DataFrames keyed by timestamp.
#
# In v1 each snapshot is solved independently -- no inter-period state coupling
# (no battery state-of-charge linkage, no ramp constraints across intervals).
# This is the simplest correct behaviour for testing the price formation logic
# and is easily extended later to add SOC tracking.

from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, ConfigDict

from nem_solver.schemas import DispatchResult, DispatchSnapshot
from nem_solver.solver import solve_dispatch


class TimeseriesResult(BaseModel):
    # arbitrary_types_allowed lets pydantic store pandas DataFrames directly
    # without trying to serialise/validate them as model fields.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Wide-form: one row per timestamp, one column per region.
    prices: pd.DataFrame
    # Long-form: one row per (timestamp, unit_id) -- easier to filter / pivot later.
    dispatch: pd.DataFrame
    storage_dispatch: pd.DataFrame
    flows: pd.DataFrame
    # Raw per-period DispatchResult objects, kept for diagnostics. Cheap because
    # they're frozen pydantic models -- no underlying LP state retained.
    raw: list[DispatchResult]


def run_timeseries(
    snapshots: Iterable[DispatchSnapshot],
    *,
    resolution_minutes: int = 5,
    solver_name: str = "highs",
) -> TimeseriesResult:
    """Solve a sequence of single-period dispatches and aggregate the outputs.

    Each snapshot is solved independently (v1 behaviour). The four output
    DataFrames make it easy to plot prices, dispatch stacks, and IC flows over
    a day without further reshaping.
    """
    raw: list[DispatchResult] = []
    # Build up rows incrementally and convert to DataFrames at the end -- much
    # cheaper than appending to DataFrames inside the loop.
    price_rows: list[dict[str, float]] = []
    dispatch_rows: list[dict[str, object]] = []
    storage_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    for snap in snapshots:
        result = solve_dispatch(
            snap, resolution_minutes=resolution_minutes, solver_name=solver_name
        )
        raw.append(result)

        ts = result.timestamp
        # One row per timestamp; columns are region IDs after the unpacking below.
        price_rows.append({"timestamp": ts, **result.rrp_by_region})

        # Long-form dispatch: each (timestamp, unit_id) -> dispatched_mw.
        for u in result.unit_dispatch:
            dispatch_rows.append(
                {"timestamp": ts, "unit_id": u.unit_id, "dispatched_mw": u.dispatched_mw}
            )

        for s in result.storage_dispatch:
            storage_rows.append(
                {
                    "timestamp": ts,
                    "unit_id": s.unit_id,
                    "discharge_mw": s.discharge_mw,
                    "charge_mw": s.charge_mw,
                    "net_mw": s.net_mw,
                }
            )

        for f in result.interconnector_flows:
            flow_rows.append(
                {
                    "timestamp": ts,
                    "ic_id": f.ic_id,
                    "flow_mw": f.flow_mw,
                    "losses_mw": f.losses_mw,
                }
            )

    # Wide prices DataFrame with timestamp as the index for easy slicing/plotting.
    prices = pd.DataFrame(price_rows).set_index("timestamp") if price_rows else pd.DataFrame()
    dispatch = pd.DataFrame(dispatch_rows)
    storage = pd.DataFrame(storage_rows)
    flows = pd.DataFrame(flow_rows)

    return TimeseriesResult(
        prices=prices,
        dispatch=dispatch,
        storage_dispatch=storage,
        flows=flows,
        raw=raw,
    )
