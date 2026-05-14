# Core LP formulation: turns a DispatchSnapshot into a linopy Model, solves it,
# and extracts a DispatchResult. This is the mathematical heart of the project.
#
# ===================== High-level structure =====================
# Decision variables (all continuous, all >= 0):
#   gen_band [g, b]   MW dispatched in band b of generator g, capped at band quantity
#   dis_band [s, b]   MW discharged in band b of storage unit s
#   chg_band [s, b]   MW charged in band b of storage unit s
#   flow_fwd [k]      MW leaving from_region toward to_region on interconnector k
#   flow_rev [k]      MW leaving to_region toward from_region on interconnector k
#   seg_fwd  [k, l]   MW assigned to forward PWL segment l of interconnector k
#   seg_rev  [k, l]   MW assigned to reverse PWL segment l of interconnector k
#
# Constraints:
#   PWL partition:   for each IC and direction, sum of segment vars == directional flow
#   Energy balance:  for each region, supply (gen + storage discharge + IC inflow
#                    minus IC losses) - (storage charge + IC outflow) == regional demand
#
# Objective (minimise):
#   sum_{g,b}  price_gen[g,b]  * gen_band[g,b]
#   + sum_{s,b} price_dis[s,b] * dis_band[s,b]
#   - sum_{s,b} price_chg[s,b] * chg_band[s,b]   <- charge has NEGATIVE coefficient
#   all multiplied by resolution_hours so the objective is in dollars
#
# RRP (regional reference price) per region = dual variable of that region's
# energy balance constraint, scaled back from $-per-resolution-period to $/MWh.
#
# ===================== Why no binaries are needed =====================
# Two places where one might expect a binary:
#
# 1. PWL loss segments. We require MLFs to be non-decreasing across segments
#    (convex losses). When the LP minimises cost, filling a low-MLF segment is
#    always cheaper than filling a high-MLF segment, so the optimal solution
#    automatically fills segments in order without any logical constraint.
#
# 2. Bidirectional storage simultaneous charge+discharge. With sane bid stacks
#    (charge_price < discharge_price), any solution with both > 0 can be
#    improved by netting them, so the LP optimum has chg * dis == 0 even
#    without a binary forbidding it.
#
# Both reductions break if you violate the assumptions, so we validate them
# in schemas.py and add tests that catch regressions.
#
# ===================== Linopy patterns used =====================
# linopy variables are xarray DataArrays of variable labels. To build a
# region-indexed energy-balance constraint we:
#   - For each region, collect the relevant 1-D variable slices via .isel()
#   - Sum each slice and merge per-region scalar expressions into a single
#     region-indexed linexpr using linopy.expressions.merge(..., dim="region")
# This way a single add_constraints call produces a constraint per region with
# duals available as a single xarray DataArray (one entry per region).

from dataclasses import dataclass, field

import linopy
import numpy as np
import pandas as pd
import xarray as xr
from linopy.expressions import merge as merge_exprs

from nem_solver.schemas import (
    DispatchResult,
    DispatchSnapshot,
    InterconnectorFlow,
    StorageDispatch,
    UnitDispatch,
)

# ---------------------------------------------------------------------------
# Internal bookkeeping helpers.
#
# As we walk the snapshot's units we flatten everything into 1-D arrays:
# every entry corresponds to one band (or one segment, or one IC). The four
# parallel lists below let us, for any flat index i, look up which unit it
# belongs to, which region, etc. This keeps the LP variables as simple 1-D
# DataArrays and pushes the unit/region groupwise logic into Python loops
# at constraint-building time.
# ---------------------------------------------------------------------------


@dataclass
class _BandIndex:
    """Flat per-band metadata. One entry per band-of-a-unit (or per IC for flows)."""
    unit_ids: list[str] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)
    band_idx: list[int] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    quantities: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.unit_ids)


@dataclass
class _SegmentIndex:
    """Flat per-segment metadata for PWL loss segments."""
    ic_ids: list[str] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)  # the receiving region
    seg_idx: list[int] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    mlfs: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ic_ids)


class DispatchModel:
    """Single-period dispatch LP for the regional NEM."""

    def __init__(self, snapshot: DispatchSnapshot, resolution_minutes: int = 5) -> None:
        if resolution_minutes <= 0:
            raise ValueError(f"resolution_minutes must be positive, got {resolution_minutes}")
        self.snapshot = snapshot
        # The dispatch LP works in MW-instantaneous; to convert price ($/MWh) and
        # MW into dollars we scale the objective by interval length in hours.
        # (5-min interval -> 1/12 h, 30-min -> 1/2 h.) This also affects the dual
        # units, which we undo in _extract_result so RRPs come back in $/MWh.
        self.resolution_hours = resolution_minutes / 60.0
        self.resolution_minutes = resolution_minutes

        self.model = linopy.Model()
        self._region_ids: list[str] = [r.region_id for r in snapshot.regions]
        self._demand_by_region: dict[str, float] = {
            r.region_id: r.demand_mw for r in snapshot.regions
        }

        # Flat indexes -- populated as we add variables.
        self._gen = _BandIndex()
        self._dis = _BandIndex()
        self._chg = _BandIndex()
        self._fwd = _BandIndex()      # quantities here are forward IC limits
        self._rev = _BandIndex()      # reverse IC limits
        self._fwd_seg = _SegmentIndex()
        self._rev_seg = _SegmentIndex()

        self._build()

    def _build(self) -> None:
        """Construct the full LP. Order matters only because variables must
        exist before constraints / objective reference them."""
        self._add_generator_vars()
        self._add_storage_vars()
        self._add_interconnector_vars()
        self._add_energy_balance()
        self._set_objective()

    # ------------------------------------------------------------------
    # Variable construction. Each block flattens (unit, band) -> a single
    # contiguous 1-D variable, with parallel metadata lists for later
    # group-by-region lookups.
    # ------------------------------------------------------------------

    def _add_generator_vars(self) -> None:
        # Walk every (generator, band) pair and append to the flat index.
        for g in self.snapshot.generators:
            for b_idx, band in enumerate(g.bands):
                self._gen.unit_ids.append(g.unit_id)
                self._gen.region_ids.append(g.region_id)
                self._gen.band_idx.append(b_idx)
                self._gen.prices.append(band.price)
                self._gen.quantities.append(band.quantity)
        if len(self._gen) > 0:
            # Variable upper bound = each band's quantity. Lower bound = 0.
            # gen_band_idx is just a 0..N-1 enumeration of all bands.
            upper = xr.DataArray(
                np.array(self._gen.quantities, dtype=float),
                coords=[("gen_band_idx", np.arange(len(self._gen)))],
            )
            self.model.add_variables(lower=0.0, upper=upper, name="gen_band")

    def _add_storage_vars(self) -> None:
        # Two parallel structures per bidirectional unit: one for discharge bands
        # (priced like generation), one for charge bands (priced like willing-to-pay
        # consumption). They become two separate LP variables but share the same
        # unit_id so the result extraction can recombine them.
        for s in self.snapshot.bidirectional_units:
            for b_idx, band in enumerate(s.discharge_bands):
                self._dis.unit_ids.append(s.unit_id)
                self._dis.region_ids.append(s.region_id)
                self._dis.band_idx.append(b_idx)
                self._dis.prices.append(band.price)
                self._dis.quantities.append(band.quantity)
            for b_idx, band in enumerate(s.charge_bands):
                self._chg.unit_ids.append(s.unit_id)
                self._chg.region_ids.append(s.region_id)
                self._chg.band_idx.append(b_idx)
                self._chg.prices.append(band.price)
                self._chg.quantities.append(band.quantity)
        if len(self._dis) > 0:
            upper = xr.DataArray(
                np.array(self._dis.quantities, dtype=float),
                coords=[("dis_band_idx", np.arange(len(self._dis)))],
            )
            self.model.add_variables(lower=0.0, upper=upper, name="dis_band")
        if len(self._chg) > 0:
            upper = xr.DataArray(
                np.array(self._chg.quantities, dtype=float),
                coords=[("chg_band_idx", np.arange(len(self._chg)))],
            )
            self.model.add_variables(lower=0.0, upper=upper, name="chg_band")

    def _add_interconnector_vars(self) -> None:
        # Each IC contributes:
        #   1 entry in self._fwd (forward directional flow variable)
        #   1 entry in self._rev (reverse directional flow variable)
        #   N entries in self._fwd_seg (forward PWL segment variables)
        #   M entries in self._rev_seg (reverse PWL segment variables)
        #
        # `region_ids` on the segment indexes records the *receiving* region --
        # losses are subtracted from the receiver's energy balance, not the sender's.
        for ic in self.snapshot.interconnectors:
            # Forward flow: sender = from_region.
            self._fwd.unit_ids.append(ic.ic_id)
            self._fwd.region_ids.append(ic.from_region)
            self._fwd.quantities.append(ic.forward_limit_mw)
            # Reverse flow: sender = to_region.
            self._rev.unit_ids.append(ic.ic_id)
            self._rev.region_ids.append(ic.to_region)
            self._rev.quantities.append(ic.reverse_limit_mw)
            # Segments. Forward losses land in the to_region (receiver of forward flow).
            for s_idx, seg in enumerate(ic.forward_loss_segments):
                self._fwd_seg.ic_ids.append(ic.ic_id)
                self._fwd_seg.region_ids.append(ic.to_region)
                self._fwd_seg.seg_idx.append(s_idx)
                self._fwd_seg.widths.append(seg.flow_to_mw - seg.flow_from_mw)
                self._fwd_seg.mlfs.append(seg.marginal_loss_factor)
            for s_idx, seg in enumerate(ic.reverse_loss_segments):
                self._rev_seg.ic_ids.append(ic.ic_id)
                self._rev_seg.region_ids.append(ic.from_region)
                self._rev_seg.seg_idx.append(s_idx)
                self._rev_seg.widths.append(seg.flow_to_mw - seg.flow_from_mw)
                self._rev_seg.mlfs.append(seg.marginal_loss_factor)

        if len(self._fwd) > 0:
            # Directional flow variables, capped at the directional MW limits.
            fwd_upper = xr.DataArray(
                np.array(self._fwd.quantities, dtype=float),
                coords=[("fwd_idx", np.arange(len(self._fwd)))],
            )
            self.model.add_variables(lower=0.0, upper=fwd_upper, name="flow_fwd")
            rev_upper = xr.DataArray(
                np.array(self._rev.quantities, dtype=float),
                coords=[("rev_idx", np.arange(len(self._rev)))],
            )
            self.model.add_variables(lower=0.0, upper=rev_upper, name="flow_rev")

            # Segment variables, each capped at its segment width.
            seg_fwd_upper = xr.DataArray(
                np.array(self._fwd_seg.widths, dtype=float),
                coords=[("fwd_seg_idx", np.arange(len(self._fwd_seg)))],
            )
            self.model.add_variables(lower=0.0, upper=seg_fwd_upper, name="seg_fwd")
            seg_rev_upper = xr.DataArray(
                np.array(self._rev_seg.widths, dtype=float),
                coords=[("rev_seg_idx", np.arange(len(self._rev_seg)))],
            )
            self.model.add_variables(lower=0.0, upper=seg_rev_upper, name="seg_rev")

            # Tie segments to their parent flow variable: sum_l seg[k,l] == flow[k].
            self._add_pwl_partition_constraints()

    def _add_pwl_partition_constraints(self) -> None:
        """Each directional flow must equal the sum of its PWL segment activations.

        Because MLFs are non-decreasing per direction (validated in schemas.py),
        cost minimisation makes the LP fill segment 0 fully before opening segment 1,
        and so on. So we don't need integer / SOS2 logic to enforce ordering.
        """
        flow_fwd = self.model.variables["flow_fwd"]
        flow_rev = self.model.variables["flow_rev"]
        seg_fwd = self.model.variables["seg_fwd"]
        seg_rev = self.model.variables["seg_rev"]

        # Forward direction: build one (sum_l seg_fwd[ic, l]) - flow_fwd[ic] expr per IC,
        # then merge into a single IC-indexed constraint.
        fwd_lhs_per_ic = []
        for ic_idx, ic_id in enumerate(self._fwd.unit_ids):
            seg_indices = [i for i, x in enumerate(self._fwd_seg.ic_ids) if x == ic_id]
            seg_sum = seg_fwd.isel(fwd_seg_idx=seg_indices).sum()
            fwd_lhs_per_ic.append(seg_sum - flow_fwd.isel(fwd_idx=ic_idx))
        if fwd_lhs_per_ic:
            combined = merge_exprs(fwd_lhs_per_ic, dim="ic")
            combined = combined.assign_coords(ic=self._fwd.unit_ids)
            rhs = xr.DataArray(np.zeros(len(self._fwd)), coords=[("ic", self._fwd.unit_ids)])
            self.model.add_constraints(combined == rhs, name="pwl_partition_fwd")

        # Reverse direction -- same pattern.
        rev_lhs_per_ic = []
        for ic_idx, ic_id in enumerate(self._rev.unit_ids):
            seg_indices = [i for i, x in enumerate(self._rev_seg.ic_ids) if x == ic_id]
            seg_sum = seg_rev.isel(rev_seg_idx=seg_indices).sum()
            rev_lhs_per_ic.append(seg_sum - flow_rev.isel(rev_idx=ic_idx))
        if rev_lhs_per_ic:
            combined = merge_exprs(rev_lhs_per_ic, dim="ic")
            combined = combined.assign_coords(ic=self._rev.unit_ids)
            rhs = xr.DataArray(np.zeros(len(self._rev)), coords=[("ic", self._rev.unit_ids)])
            self.model.add_constraints(combined == rhs, name="pwl_partition_rev")

    def _add_energy_balance(self) -> None:
        """For each region: supply == demand. The dual of this constraint = RRP.

        Per region r, the LHS (in MW) is:
            local generation
          + local storage discharge
          - local storage charge
          + interconnector inflow (forward flows arriving here as receiver,
            plus reverse flows arriving here as receiver)
          - interconnector outflow (forward flows leaving here as sender,
            plus reverse flows leaving here as sender)
          - losses on inflows (segment-by-segment, weighted by MLF)
          == demand[r]
        """
        # Pull variable handles once. None values mean "no such variable in this model".
        v = self.model.variables
        gen = v["gen_band"] if len(self._gen) else None
        dis = v["dis_band"] if len(self._dis) else None
        chg = v["chg_band"] if len(self._chg) else None
        flow_fwd = v["flow_fwd"] if len(self._fwd) else None
        flow_rev = v["flow_rev"] if len(self._rev) else None
        seg_fwd = v["seg_fwd"] if len(self._fwd_seg) else None
        seg_rev = v["seg_rev"] if len(self._rev_seg) else None

        per_region_exprs = []
        rhs_values = []
        for r in self._region_ids:
            terms: list = []  # additive contributions to this region's LHS

            # 1) Local generators contribute the sum of their dispatched bands.
            if gen is not None:
                idx = [i for i, rg in enumerate(self._gen.region_ids) if rg == r]
                if idx:
                    terms.append(gen.isel(gen_band_idx=idx).sum())

            # 2) Local storage discharge adds to supply.
            if dis is not None:
                idx = [i for i, rg in enumerate(self._dis.region_ids) if rg == r]
                if idx:
                    terms.append(dis.isel(dis_band_idx=idx).sum())

            # 3) Local storage charge subtracts from supply (it's a load).
            if chg is not None:
                idx = [i for i, rg in enumerate(self._chg.region_ids) if rg == r]
                if idx:
                    terms.append(-1.0 * chg.isel(chg_band_idx=idx).sum())

            # 4) Forward IC flow:
            #    - if r is the from_region, we SUBTRACT flow_fwd (it leaves r)
            #    - if r is the to_region, we ADD flow_fwd and SUBTRACT forward losses
            if flow_fwd is not None:
                send_idx = [
                    i
                    for i, ic in enumerate(self.snapshot.interconnectors)
                    if ic.from_region == r
                ]
                if send_idx:
                    terms.append(-1.0 * flow_fwd.isel(fwd_idx=send_idx).sum())
                recv_idx = [
                    i
                    for i, ic in enumerate(self.snapshot.interconnectors)
                    if ic.to_region == r
                ]
                if recv_idx:
                    terms.append(flow_fwd.isel(fwd_idx=recv_idx).sum())
                # Forward losses are an expression over the segment vars: sum_l mlf_l * seg_fwd[k,l]
                # for ICs whose receiver is r. We construct the MLF DataArray indexed by the
                # selected segment positions and dot-multiply with the segment slice.
                if seg_fwd is not None:
                    seg_recv = [
                        i for i, rg in enumerate(self._fwd_seg.region_ids) if rg == r
                    ]
                    if seg_recv:
                        mlf_arr = xr.DataArray(
                            np.array([self._fwd_seg.mlfs[i] for i in seg_recv], dtype=float),
                            coords=[("fwd_seg_idx", np.array(seg_recv))],
                        )
                        terms.append(-1.0 * (seg_fwd.isel(fwd_seg_idx=seg_recv) * mlf_arr).sum())

            # 5) Reverse IC flow: mirror of forward. The reverse flow leaves to_region
            #    and arrives in from_region, with reverse-direction losses.
            if flow_rev is not None:
                send_idx = [
                    i
                    for i, ic in enumerate(self.snapshot.interconnectors)
                    if ic.to_region == r
                ]
                if send_idx:
                    terms.append(-1.0 * flow_rev.isel(rev_idx=send_idx).sum())
                recv_idx = [
                    i
                    for i, ic in enumerate(self.snapshot.interconnectors)
                    if ic.from_region == r
                ]
                if recv_idx:
                    terms.append(flow_rev.isel(rev_idx=recv_idx).sum())
                if seg_rev is not None:
                    seg_recv = [
                        i for i, rg in enumerate(self._rev_seg.region_ids) if rg == r
                    ]
                    if seg_recv:
                        mlf_arr = xr.DataArray(
                            np.array([self._rev_seg.mlfs[i] for i in seg_recv], dtype=float),
                            coords=[("rev_seg_idx", np.array(seg_recv))],
                        )
                        terms.append(-1.0 * (seg_rev.isel(rev_seg_idx=seg_recv) * mlf_arr).sum())

            if not terms:
                # A region with no generators, no storage, and no interconnectors
                # has no way to meet demand. Catch this clearly here rather than
                # producing a silently infeasible LP.
                raise ValueError(f"Region {r} has no generators, storage, or interconnectors")

            # Sum all term contributions into a single linexpr for this region.
            lhs_r = terms[0]
            for t in terms[1:]:
                lhs_r = lhs_r + t
            per_region_exprs.append(lhs_r)
            rhs_values.append(self._demand_by_region[r])

        # Stack the per-region scalar linexprs into a single region-indexed linexpr,
        # so the constraint produces one row per region with duals available as
        # a single xarray DataArray indexed by region.
        combined = merge_exprs(per_region_exprs, dim="region")
        combined = combined.assign_coords(region=self._region_ids)
        rhs = xr.DataArray(
            np.array(rhs_values, dtype=float), coords=[("region", self._region_ids)]
        )
        self.model.add_constraints(combined == rhs, name="energy_balance")

    def _set_objective(self) -> None:
        """Total cost across all dispatched bands, scaled to dollars over the period.

        Generation and storage discharge add positive cost. Storage charge has a
        NEGATIVE coefficient (-price): paying $P/MWh to charge means the LP is
        $P/MWh better off when it can charge cheaply. Result: the LP charges
        whenever the regional price RRP < charge bid, exactly the desired economic
        behaviour for a battery deciding when to consume.
        """
        terms = []
        if len(self._gen):
            gen = self.model.variables["gen_band"]
            prices = xr.DataArray(
                np.array(self._gen.prices, dtype=float),
                coords=[("gen_band_idx", np.arange(len(self._gen)))],
            )
            terms.append((gen * prices).sum())
        if len(self._dis):
            dis = self.model.variables["dis_band"]
            prices = xr.DataArray(
                np.array(self._dis.prices, dtype=float),
                coords=[("dis_band_idx", np.arange(len(self._dis)))],
            )
            terms.append((dis * prices).sum())
        if len(self._chg):
            chg = self.model.variables["chg_band"]
            # Negate prices: charging at price P contributes -P*MW to the cost.
            prices = xr.DataArray(
                np.array(self._chg.prices, dtype=float) * -1.0,
                coords=[("chg_band_idx", np.arange(len(self._chg)))],
            )
            terms.append((chg * prices).sum())
        if not terms:
            raise ValueError("No dispatchable units; objective is empty")
        obj = terms[0]
        for t in terms[1:]:
            obj = obj + t
        # Multiplying by resolution_hours converts $/MWh * MW (a power-cost rate)
        # into actual dollars over the dispatch interval. This is what makes the
        # objective and the duals dimensionally meaningful.
        self.model.add_objective(self.resolution_hours * obj)

    # ------------------------------------------------------------------
    # Solve & result extraction.
    # ------------------------------------------------------------------

    def solve(self, solver_name: str = "highs") -> DispatchResult:
        # linopy.solve() returns (termination_status, condition); we only use the first.
        # HiGHS is the default open-source LP solver; it's installed as a Python wheel
        # via the highspy dependency.
        status, _ = self.model.solve(solver_name=solver_name)
        return self._extract_result(status)

    def _extract_result(self, status: str) -> DispatchResult:
        if status != "ok":
            # We don't try to recover from infeasibility / unboundedness here --
            # a failed solve in this context indicates malformed inputs.
            raise RuntimeError(f"Solver did not return ok status: {status}")

        # ---- Generator dispatch ------------------------------------------------
        # Sum band-level dispatch back up to per-unit totals.
        unit_dispatch: list[UnitDispatch] = []
        if len(self._gen):
            gen_sol = self.model.variables["gen_band"].solution.values
            df = pd.DataFrame(
                {
                    "unit_id": self._gen.unit_ids,
                    "band_idx": self._gen.band_idx,
                    "mw": gen_sol,
                }
            )
            for unit_id, group in df.groupby("unit_id", sort=False):
                bands = tuple(
                    group.sort_values("band_idx")["mw"].astype(float).tolist()
                )
                unit_dispatch.append(
                    UnitDispatch(
                        unit_id=str(unit_id),
                        dispatched_mw=float(group["mw"].sum()),
                        band_dispatch_mw=bands,
                    )
                )

        # ---- Storage dispatch --------------------------------------------------
        # Combine charge and discharge per unit. net_mw is the unit's effect on
        # its region's energy balance: positive = net injection, negative = net load.
        storage_dispatch: list[StorageDispatch] = []
        if len(self._dis) or len(self._chg):
            dis_sol = (
                self.model.variables["dis_band"].solution.values if len(self._dis) else np.array([])
            )
            chg_sol = (
                self.model.variables["chg_band"].solution.values if len(self._chg) else np.array([])
            )
            unit_ids = sorted({*self._dis.unit_ids, *self._chg.unit_ids})
            for u in unit_ids:
                dis_idx = [i for i, x in enumerate(self._dis.unit_ids) if x == u]
                chg_idx = [i for i, x in enumerate(self._chg.unit_ids) if x == u]
                dis_bands = tuple(float(dis_sol[i]) for i in dis_idx)
                chg_bands = tuple(float(chg_sol[i]) for i in chg_idx)
                d = float(sum(dis_bands))
                c = float(sum(chg_bands))
                storage_dispatch.append(
                    StorageDispatch(
                        unit_id=u,
                        discharge_mw=d,
                        charge_mw=c,
                        net_mw=d - c,
                        discharge_band_mw=dis_bands,
                        charge_band_mw=chg_bands,
                    )
                )

        # ---- Interconnector flows ----------------------------------------------
        # Combine fwd + rev into a single signed flow (+ = forward direction) and
        # sum segment activations to recover total losses.
        ic_flows: list[InterconnectorFlow] = []
        if len(self._fwd):
            fwd_sol = self.model.variables["flow_fwd"].solution.values
            rev_sol = self.model.variables["flow_rev"].solution.values
            seg_fwd_sol = self.model.variables["seg_fwd"].solution.values
            seg_rev_sol = self.model.variables["seg_rev"].solution.values
            for i, ic in enumerate(self.snapshot.interconnectors):
                fwd_flow = float(fwd_sol[i])
                rev_flow = float(rev_sol[i])
                # losses = sum_l (mlf_l * seg_l) for this IC and direction.
                fwd_loss = float(
                    sum(
                        seg_fwd_sol[j] * self._fwd_seg.mlfs[j]
                        for j, ic_id in enumerate(self._fwd_seg.ic_ids)
                        if ic_id == ic.ic_id
                    )
                )
                rev_loss = float(
                    sum(
                        seg_rev_sol[j] * self._rev_seg.mlfs[j]
                        for j, ic_id in enumerate(self._rev_seg.ic_ids)
                        if ic_id == ic.ic_id
                    )
                )
                signed_flow = fwd_flow - rev_flow  # + = from_region -> to_region
                losses = fwd_loss + rev_loss       # only one is nonzero in practice
                # sent_out = MW leaving the source region; received = MW arriving
                # at the sink (= sent_out - losses). Sign of signed_flow tells us
                # which direction is active.
                sent_out = fwd_flow if signed_flow >= 0 else rev_flow
                received = sent_out - losses
                ic_flows.append(
                    InterconnectorFlow(
                        ic_id=ic.ic_id,
                        flow_mw=signed_flow,
                        losses_mw=losses,
                        sent_out_mw=sent_out,
                        received_mw=received,
                    )
                )

        # ---- Regional reference prices (RRPs) ----------------------------------
        # The dual of region r's energy balance is the marginal cost of supplying
        # one extra MWh of demand in region r -- exactly the RRP. Because we
        # scaled the objective by resolution_hours, the dual is in $-per-period;
        # divide by resolution_hours to get $/MWh.
        duals = self.model.constraints["energy_balance"].dual
        rrp_by_region = {
            str(r): float(duals.sel(region=r).item()) / self.resolution_hours
            for r in self._region_ids
        }

        return DispatchResult(
            timestamp=self.snapshot.timestamp,
            objective_value=float(self.model.objective.value),
            rrp_by_region=rrp_by_region,
            unit_dispatch=tuple(unit_dispatch),
            storage_dispatch=tuple(storage_dispatch),
            interconnector_flows=tuple(ic_flows),
            solver_status=status,
        )
