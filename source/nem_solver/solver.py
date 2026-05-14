# Thin functional wrapper around DispatchModel. Most callers should use this --
# the class is exported only for power users who want to inspect the LP, mutate
# parameters, or eventually warm-start across periods.

from nem_solver.model import DispatchModel
from nem_solver.schemas import DispatchResult, DispatchSnapshot


def solve_dispatch(
    snapshot: DispatchSnapshot,
    resolution_minutes: int = 5,
    *,
    solver_name: str = "highs",
) -> DispatchResult:
    """Solve a single-period regional dispatch LP and return prices and dispatched MW.

    Parameters
    ----------
    snapshot:
        Inputs for one dispatch interval (regions, units, interconnectors).
    resolution_minutes:
        Length of the dispatch interval. 5 for real-time NEM dispatch, 30 for
        pre-dispatch / settlement. Affects only the objective scaling -- the
        underlying LP structure is identical.
    solver_name:
        Any LP solver linopy can drive. HiGHS is the default and is bundled.
    """
    return DispatchModel(snapshot, resolution_minutes=resolution_minutes).solve(
        solver_name=solver_name
    )
