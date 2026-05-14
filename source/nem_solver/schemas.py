# Input and output data models for the dispatch solver.
#
# All types are pydantic v2 BaseModels with frozen=True so they're hashable and
# safe to share. extra="forbid" means typos in field names raise instead of being
# silently ignored. Cross-field invariants (e.g. band prices must be monotonic,
# IC endpoints must reference declared regions) are enforced via @model_validator
# hooks so a malformed snapshot fails *before* it reaches the LP builder.

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nem_solver.constants import RegionId


class _Frozen(BaseModel):
    # Shared base config: immutable instances + reject unknown fields.
    model_config = ConfigDict(frozen=True, extra="forbid")


class BidBand(_Frozen):
    """One (price, quantity) tuple within a generator's or storage unit's bid stack.

    A NEMDE-style bid stack has up to 10 of these; the LP fills lower-priced bands
    before higher-priced ones to minimise dispatch cost.
    """
    price: float = Field(description="Bid price in $/MWh")
    quantity: float = Field(ge=0.0, description="Capacity available in this band, MW")


class Region(_Frozen):
    # One pricing region in this snapshot. demand_mw is the operational demand
    # the LP must serve (no unserved-energy slack in v1, so demand is hard).
    region_id: RegionId
    demand_mw: float = Field(ge=0.0)


class Generator(_Frozen):
    """A scheduled generator with a multi-band offer stack."""
    unit_id: str
    region_id: RegionId
    bands: tuple[BidBand, ...]
    max_capacity_mw: float = Field(ge=0.0)
    min_capacity_mw: float = Field(default=0.0, ge=0.0)  # not used in v1; placeholder
    # Reserved for future FCAS co-optimisation: each FCAS service (e.g. "raise6sec")
    # would carry its own bid stack. Untouched by the v1 energy-only LP.
    fcas_offers: dict[str, tuple[BidBand, ...]] = Field(
        default_factory=dict,
        description="Reserved for future FCAS co-optimisation; unused in v1.",
    )

    @model_validator(mode="after")
    def _check_bands(self) -> Self:
        # 1) Must have at least one band -- otherwise the unit can't be dispatched.
        if len(self.bands) == 0:
            raise ValueError(f"Generator {self.unit_id} must have at least one bid band")
        # 2) Bid prices must be non-decreasing across the stack: band 0 cheapest,
        #    band N-1 most expensive. This matches NEMDE convention and means the
        #    LP can fill bands in index order without needing SOS / binary logic.
        prices = [b.price for b in self.bands]
        if any(p2 < p1 for p1, p2 in zip(prices, prices[1:], strict=False)):
            raise ValueError(
                f"Generator {self.unit_id} bid band prices must be non-decreasing, got {prices}"
            )
        # 3) Sum of band quantities must equal max_capacity_mw -- catches data
        #    entry errors where someone forgets to align the stack with the cap.
        total_q = sum(b.quantity for b in self.bands)
        if abs(total_q - self.max_capacity_mw) > 1e-6:
            raise ValueError(
                f"Generator {self.unit_id} band quantities sum to {total_q} but "
                f"max_capacity_mw is {self.max_capacity_mw}"
            )
        if self.min_capacity_mw > self.max_capacity_mw:
            raise ValueError(
                f"Generator {self.unit_id} min_capacity_mw {self.min_capacity_mw} > "
                f"max_capacity_mw {self.max_capacity_mw}"
            )
        return self


class BidirectionalUnit(_Frozen):
    """A battery or pumped-hydro unit with separate discharge (gen) and charge (load) bids.

    Two bid stacks because the unit acts as a generator when discharging and as a
    consumer when charging. In the LP, charge variables get a *negative* objective
    coefficient -- charging at price P reduces total dispatch cost by P per MWh,
    representing the unit's willingness to pay to consume. The LP will charge
    whenever the regional price RRP < charge band price.
    """
    unit_id: str
    region_id: RegionId
    discharge_bands: tuple[BidBand, ...]
    charge_bands: tuple[BidBand, ...]
    max_discharge_mw: float = Field(ge=0.0)
    max_charge_mw: float = Field(ge=0.0)
    # Round-trip efficiency is reserved for future SOC tracking; in v1 each period
    # is solved independently so efficiency only matters across periods.
    round_trip_efficiency: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_bands(self) -> Self:
        if len(self.discharge_bands) == 0 or len(self.charge_bands) == 0:
            raise ValueError(
                f"BidirectionalUnit {self.unit_id} must have at least one band per direction"
            )
        # Discharge bands behave like generator bands -- prices non-decreasing
        # (cheapest band fills first when LP minimises cost).
        dis_prices = [b.price for b in self.discharge_bands]
        if any(p2 < p1 for p1, p2 in zip(dis_prices, dis_prices[1:], strict=False)):
            raise ValueError(
                f"BidirectionalUnit {self.unit_id} discharge prices must be non-decreasing"
            )
        # Charge bands are inverted: highest willingness-to-pay first. The LP gives
        # charge variables coefficient -price, so the band with the largest price
        # has the most negative coefficient and fills first. Storing them in
        # descending order keeps band index 0 = "most desirable to fill".
        chg_prices = [b.price for b in self.charge_bands]
        if any(p2 > p1 for p1, p2 in zip(chg_prices, chg_prices[1:], strict=False)):
            raise ValueError(
                f"BidirectionalUnit {self.unit_id} charge prices must be non-increasing "
                "(highest willingness-to-pay band first)"
            )
        # Stack quantities must align with the unit's directional capacity.
        dis_total = sum(b.quantity for b in self.discharge_bands)
        if abs(dis_total - self.max_discharge_mw) > 1e-6:
            raise ValueError(
                f"BidirectionalUnit {self.unit_id} discharge band quantities sum to {dis_total} "
                f"but max_discharge_mw is {self.max_discharge_mw}"
            )
        chg_total = sum(b.quantity for b in self.charge_bands)
        if abs(chg_total - self.max_charge_mw) > 1e-6:
            raise ValueError(
                f"BidirectionalUnit {self.unit_id} charge band quantities sum to {chg_total} "
                f"but max_charge_mw is {self.max_charge_mw}"
            )
        return self


class PWLLossSegment(_Frozen):
    """One linear piece of an interconnector's loss curve.

    Real interconnector losses are roughly quadratic in flow: loss ~= a*F + b*F^2.
    We approximate the curve with several linear segments. Within each segment,
    losses grow linearly at rate `marginal_loss_factor` per MW of flow.

    For convex loss curves (real-world interconnectors), MLFs are non-decreasing
    across segments, which lets the LP avoid binary variables: minimising cost
    naturally fills the lowest-MLF segment before moving on to higher-MLF ones.
    """
    flow_from_mw: float = Field(ge=0.0)
    flow_to_mw: float = Field(ge=0.0)
    marginal_loss_factor: float = Field(
        ge=0.0, description="Incremental losses per MW on this segment (dimensionless)"
    )

    @model_validator(mode="after")
    def _check_widths(self) -> Self:
        # Zero or negative-width segments would create degenerate LP variables.
        if self.flow_to_mw <= self.flow_from_mw:
            raise ValueError(
                f"PWL segment must have flow_to_mw > flow_from_mw, "
                f"got [{self.flow_from_mw}, {self.flow_to_mw}]"
            )
        return self


class Interconnector(_Frozen):
    """A regional transmission link with separate forward and reverse loss curves.

    AEMO models each interconnector as directional: forward = from_region -> to_region.
    Each direction has its own MW limit and its own piecewise-linear loss curve.
    We split flow into two non-negative variables (flow_fwd, flow_rev) so the loss
    approximation can be different in each direction. Cost-minimisation guarantees
    the LP won't run both directions at once for any sane snapshot.
    """
    ic_id: str
    from_region: RegionId
    to_region: RegionId
    forward_limit_mw: float = Field(ge=0.0)
    reverse_limit_mw: float = Field(ge=0.0)
    forward_loss_segments: tuple[PWLLossSegment, ...]
    reverse_loss_segments: tuple[PWLLossSegment, ...]

    @model_validator(mode="after")
    def _check_topology_and_segments(self) -> Self:
        if self.from_region == self.to_region:
            raise ValueError(f"Interconnector {self.ic_id} cannot be self-looped")
        # Both directions get the same structural checks (start at 0, contiguous,
        # MLFs non-decreasing for convexity, total coverage >= the directional limit).
        _validate_segments(self.ic_id, "forward", self.forward_loss_segments, self.forward_limit_mw)
        _validate_segments(self.ic_id, "reverse", self.reverse_loss_segments, self.reverse_limit_mw)
        return self


def _validate_segments(
    ic_id: str, direction: str, segments: tuple[PWLLossSegment, ...], limit: float
) -> None:
    """Structural checks on a directional PWL stack.

    The LP relies on these properties:
      - Segments cover [0, limit] without gaps or overlaps.
      - MLFs are non-decreasing -> losses are convex -> no binaries needed for
        segment ordering (lowest-MLF segment fills first when cost is minimised).
    """
    if len(segments) == 0:
        raise ValueError(f"Interconnector {ic_id} {direction} must have at least one PWL segment")
    if abs(segments[0].flow_from_mw) > 1e-9:
        raise ValueError(
            f"Interconnector {ic_id} {direction} segments must start at flow=0, "
            f"got {segments[0].flow_from_mw}"
        )
    # Contiguity: each segment's flow_from must equal the previous flow_to.
    for prev, curr in zip(segments, segments[1:], strict=False):
        if abs(curr.flow_from_mw - prev.flow_to_mw) > 1e-9:
            raise ValueError(
                f"Interconnector {ic_id} {direction} segments must be contiguous; "
                f"gap between {prev.flow_to_mw} and {curr.flow_from_mw}"
            )
    # Convexity check: marginal loss factor must rise (or stay flat) as flow grows.
    mlfs = [s.marginal_loss_factor for s in segments]
    if any(m2 < m1 for m1, m2 in zip(mlfs, mlfs[1:], strict=False)):
        raise ValueError(
            f"Interconnector {ic_id} {direction} MLFs must be non-decreasing for convex losses, "
            f"got {mlfs}"
        )
    # Coverage: the segments must reach at least up to the directional MW limit,
    # otherwise the LP could be infeasible if it wanted to flow up to the cap.
    total_width = segments[-1].flow_to_mw
    if total_width < limit - 1e-6:
        raise ValueError(
            f"Interconnector {ic_id} {direction} segments cover up to {total_width} MW but "
            f"limit is {limit} MW"
        )


class DispatchSnapshot(_Frozen):
    """All inputs needed to solve one dispatch interval: regions + units + interconnectors."""
    timestamp: datetime
    regions: tuple[Region, ...]
    generators: tuple[Generator, ...]
    bidirectional_units: tuple[BidirectionalUnit, ...] = ()
    interconnectors: tuple[Interconnector, ...]

    @model_validator(mode="after")
    def _check_topology(self) -> Self:
        # All cross-references between objects must resolve. These checks make
        # broken snapshots fail with a clear error rather than producing silent
        # nonsense in the LP (e.g. a generator that's not in any region balance).
        region_ids = {r.region_id for r in self.regions}
        if len(region_ids) != len(self.regions):
            raise ValueError("Duplicate region IDs in snapshot")
        for g in self.generators:
            if g.region_id not in region_ids:
                raise ValueError(f"Generator {g.unit_id} references unknown region {g.region_id}")
        for s in self.bidirectional_units:
            if s.region_id not in region_ids:
                raise ValueError(
                    f"BidirectionalUnit {s.unit_id} references unknown region {s.region_id}"
                )
        for ic in self.interconnectors:
            if ic.from_region not in region_ids:
                raise ValueError(
                    f"Interconnector {ic.ic_id} from_region {ic.from_region} not in regions"
                )
            if ic.to_region not in region_ids:
                raise ValueError(
                    f"Interconnector {ic.ic_id} to_region {ic.to_region} not in regions"
                )
        # Unit IDs must be globally unique (we use them as dictionary keys later).
        unit_ids = [g.unit_id for g in self.generators] + [
            s.unit_id for s in self.bidirectional_units
        ]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Duplicate unit IDs across generators and bidirectional units")
        ic_ids = [ic.ic_id for ic in self.interconnectors]
        if len(set(ic_ids)) != len(ic_ids):
            raise ValueError("Duplicate interconnector IDs")
        return self


# ---- Output types -----------------------------------------------------------
# These are populated after solve(); they're mostly plain data classes but
# inherit the same frozen / extra=forbid base for consistency.


class UnitDispatch(_Frozen):
    """Per-generator dispatch result. band_dispatch_mw lets you see how much
    each price band contributed -- useful for diagnostics and price-setting checks."""
    unit_id: str
    dispatched_mw: float
    band_dispatch_mw: tuple[float, ...]


class StorageDispatch(_Frozen):
    """Per-bidirectional-unit result. net_mw = discharge_mw - charge_mw is the
    region-level effect (positive when injecting, negative when consuming)."""
    unit_id: str
    discharge_mw: float
    charge_mw: float
    net_mw: float
    discharge_band_mw: tuple[float, ...]
    charge_band_mw: tuple[float, ...]


class InterconnectorFlow(_Frozen):
    """Per-interconnector result with signed flow (+ = forward direction).
    sent_out_mw is what leaves the source region; received_mw is what arrives
    at the sink (= sent_out - losses)."""
    ic_id: str
    flow_mw: float
    losses_mw: float
    sent_out_mw: float
    received_mw: float


class DispatchResult(_Frozen):
    """The full output of one dispatch solve. rrp_by_region holds the regional
    reference prices in $/MWh, recovered from the duals on the energy balance
    constraint (the "shadow price" of meeting one extra MW of demand)."""
    timestamp: datetime
    objective_value: float
    rrp_by_region: dict[str, float]
    unit_dispatch: tuple[UnitDispatch, ...]
    storage_dispatch: tuple[StorageDispatch, ...]
    interconnector_flows: tuple[InterconnectorFlow, ...]
    solver_status: str
