from datetime import datetime

import pytest
from nem_solver import (
    BidBand,
    BidirectionalUnit,
    DispatchSnapshot,
    Generator,
    Interconnector,
    PWLLossSegment,
    Region,
)


def _zero_loss_segments(limit_mw: float) -> tuple[PWLLossSegment, ...]:
    return (
        PWLLossSegment(flow_from_mw=0.0, flow_to_mw=limit_mw, marginal_loss_factor=0.0),
    )


@pytest.fixture
def single_region_snapshot() -> DispatchSnapshot:
    return DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(Region(region_id="NSW1", demand_mw=60.0),),
        generators=(
            Generator(
                unit_id="G1",
                region_id="NSW1",
                bands=(BidBand(price=50.0, quantity=100.0),),
                max_capacity_mw=100.0,
            ),
        ),
        interconnectors=(),
    )


@pytest.fixture
def stacked_bands_snapshot() -> DispatchSnapshot:
    return DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(Region(region_id="NSW1", demand_mw=50.0),),
        generators=(
            Generator(
                unit_id="G1",
                region_id="NSW1",
                bands=(
                    BidBand(price=30.0, quantity=20.0),
                    BidBand(price=60.0, quantity=40.0),
                    BidBand(price=100.0, quantity=40.0),
                ),
                max_capacity_mw=100.0,
            ),
        ),
        interconnectors=(),
    )


@pytest.fixture
def two_region_snapshot() -> DispatchSnapshot:
    """NSW (cheap) -> VIC (expensive); IC limit 200 MW (unconstrained for 100 MW demand each)."""
    return DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(
            Region(region_id="NSW1", demand_mw=100.0),
            Region(region_id="VIC1", demand_mw=100.0),
        ),
        generators=(
            Generator(
                unit_id="NSW_GEN",
                region_id="NSW1",
                bands=(BidBand(price=40.0, quantity=400.0),),
                max_capacity_mw=400.0,
            ),
            Generator(
                unit_id="VIC_GEN",
                region_id="VIC1",
                bands=(BidBand(price=120.0, quantity=400.0),),
                max_capacity_mw=400.0,
            ),
        ),
        interconnectors=(
            Interconnector(
                ic_id="NSW1-VIC1",
                from_region="NSW1",
                to_region="VIC1",
                forward_limit_mw=200.0,
                reverse_limit_mw=200.0,
                forward_loss_segments=_zero_loss_segments(200.0),
                reverse_loss_segments=_zero_loss_segments(200.0),
            ),
        ),
    )


@pytest.fixture
def two_region_constrained_snapshot() -> DispatchSnapshot:
    """Same as two_region_snapshot but IC limit 50 -- VIC has to dispatch its expensive gen."""
    return DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(
            Region(region_id="NSW1", demand_mw=100.0),
            Region(region_id="VIC1", demand_mw=100.0),
        ),
        generators=(
            Generator(
                unit_id="NSW_GEN",
                region_id="NSW1",
                bands=(BidBand(price=40.0, quantity=400.0),),
                max_capacity_mw=400.0,
            ),
            Generator(
                unit_id="VIC_GEN",
                region_id="VIC1",
                bands=(BidBand(price=120.0, quantity=400.0),),
                max_capacity_mw=400.0,
            ),
        ),
        interconnectors=(
            Interconnector(
                ic_id="NSW1-VIC1",
                from_region="NSW1",
                to_region="VIC1",
                forward_limit_mw=50.0,
                reverse_limit_mw=50.0,
                forward_loss_segments=_zero_loss_segments(50.0),
                reverse_loss_segments=_zero_loss_segments(50.0),
            ),
        ),
    )


@pytest.fixture
def battery_high_price_snapshot() -> DispatchSnapshot:
    """Single region with a very expensive marginal generator and a battery.
    Battery should discharge at maximum because RRP > discharge bid > charge bid."""
    return DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(Region(region_id="NSW1", demand_mw=350.0),),
        generators=(
            Generator(
                unit_id="CHEAP",
                region_id="NSW1",
                bands=(BidBand(price=40.0, quantity=300.0),),
                max_capacity_mw=300.0,
            ),
            Generator(
                unit_id="PEAKER",
                region_id="NSW1",
                bands=(BidBand(price=300.0, quantity=200.0),),
                max_capacity_mw=200.0,
            ),
        ),
        bidirectional_units=(
            BidirectionalUnit(
                unit_id="BATT1",
                region_id="NSW1",
                discharge_bands=(BidBand(price=200.0, quantity=50.0),),
                charge_bands=(BidBand(price=20.0, quantity=50.0),),
                max_discharge_mw=50.0,
                max_charge_mw=50.0,
            ),
        ),
        interconnectors=(),
    )


@pytest.fixture
def battery_low_price_snapshot() -> DispatchSnapshot:
    """Region with abundant cheap generation; battery should charge because charge WTP > RRP."""
    return DispatchSnapshot(
        timestamp=datetime(2026, 5, 8, 12, 0),
        regions=(Region(region_id="NSW1", demand_mw=100.0),),
        generators=(
            Generator(
                unit_id="CHEAP",
                region_id="NSW1",
                bands=(BidBand(price=10.0, quantity=300.0),),
                max_capacity_mw=300.0,
            ),
        ),
        bidirectional_units=(
            BidirectionalUnit(
                unit_id="BATT1",
                region_id="NSW1",
                discharge_bands=(BidBand(price=200.0, quantity=50.0),),
                charge_bands=(BidBand(price=50.0, quantity=50.0),),
                max_discharge_mw=50.0,
                max_charge_mw=50.0,
            ),
        ),
        interconnectors=(),
    )
