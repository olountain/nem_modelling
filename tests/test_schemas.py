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
from pydantic import ValidationError


def _segments_to(limit_mw: float, mlfs: tuple[float, ...] = (0.0,)) -> tuple[PWLLossSegment, ...]:
    n = len(mlfs)
    width = limit_mw / n
    return tuple(
        PWLLossSegment(
            flow_from_mw=i * width,
            flow_to_mw=(i + 1) * width,
            marginal_loss_factor=mlfs[i],
        )
        for i in range(n)
    )


def test_generator_band_prices_must_be_non_decreasing():
    with pytest.raises(ValidationError, match="non-decreasing"):
        Generator(
            unit_id="G1",
            region_id="NSW1",
            bands=(BidBand(price=50.0, quantity=50.0), BidBand(price=30.0, quantity=50.0)),
            max_capacity_mw=100.0,
        )


def test_generator_band_quantities_must_sum_to_max_capacity():
    with pytest.raises(ValidationError, match="sum"):
        Generator(
            unit_id="G1",
            region_id="NSW1",
            bands=(BidBand(price=30.0, quantity=40.0),),
            max_capacity_mw=100.0,
        )


def test_region_id_typo_rejected():
    with pytest.raises(ValidationError):
        Region(region_id="NSW", demand_mw=100.0)  # type: ignore[arg-type]


def test_pwl_mlfs_must_be_non_decreasing():
    with pytest.raises(ValidationError, match="non-decreasing"):
        Interconnector(
            ic_id="X",
            from_region="NSW1",
            to_region="VIC1",
            forward_limit_mw=100.0,
            reverse_limit_mw=100.0,
            forward_loss_segments=_segments_to(100.0, (0.05, 0.02)),
            reverse_loss_segments=_segments_to(100.0, (0.0,)),
        )


def test_pwl_segments_must_be_contiguous():
    with pytest.raises(ValidationError, match="contiguous"):
        Interconnector(
            ic_id="X",
            from_region="NSW1",
            to_region="VIC1",
            forward_limit_mw=100.0,
            reverse_limit_mw=100.0,
            forward_loss_segments=(
                PWLLossSegment(flow_from_mw=0.0, flow_to_mw=50.0, marginal_loss_factor=0.0),
                PWLLossSegment(flow_from_mw=60.0, flow_to_mw=100.0, marginal_loss_factor=0.05),
            ),
            reverse_loss_segments=_segments_to(100.0, (0.0,)),
        )


def test_pwl_segments_must_start_at_zero():
    with pytest.raises(ValidationError, match="must start at flow=0"):
        Interconnector(
            ic_id="X",
            from_region="NSW1",
            to_region="VIC1",
            forward_limit_mw=100.0,
            reverse_limit_mw=100.0,
            forward_loss_segments=(
                PWLLossSegment(flow_from_mw=10.0, flow_to_mw=100.0, marginal_loss_factor=0.0),
            ),
            reverse_loss_segments=_segments_to(100.0, (0.0,)),
        )


def test_interconnector_self_loop_rejected():
    with pytest.raises(ValidationError, match="self-looped"):
        Interconnector(
            ic_id="X",
            from_region="NSW1",
            to_region="NSW1",
            forward_limit_mw=100.0,
            reverse_limit_mw=100.0,
            forward_loss_segments=_segments_to(100.0),
            reverse_loss_segments=_segments_to(100.0),
        )


def test_bidirectional_unit_charge_prices_non_increasing():
    with pytest.raises(ValidationError, match="charge prices"):
        BidirectionalUnit(
            unit_id="B1",
            region_id="NSW1",
            discharge_bands=(BidBand(price=200.0, quantity=50.0),),
            charge_bands=(
                BidBand(price=10.0, quantity=25.0),
                BidBand(price=20.0, quantity=25.0),
            ),
            max_discharge_mw=50.0,
            max_charge_mw=50.0,
        )


def test_snapshot_unknown_unit_region_rejected():
    with pytest.raises(ValidationError, match="unknown region"):
        DispatchSnapshot(
            timestamp=datetime(2026, 5, 8, 12, 0),
            regions=(Region(region_id="NSW1", demand_mw=100.0),),
            generators=(
                Generator(
                    unit_id="G1",
                    region_id="VIC1",
                    bands=(BidBand(price=30.0, quantity=100.0),),
                    max_capacity_mw=100.0,
                ),
            ),
            interconnectors=(),
        )


def test_snapshot_duplicate_unit_id_rejected():
    with pytest.raises(ValidationError, match="Duplicate unit IDs"):
        DispatchSnapshot(
            timestamp=datetime(2026, 5, 8, 12, 0),
            regions=(Region(region_id="NSW1", demand_mw=50.0),),
            generators=(
                Generator(
                    unit_id="G1",
                    region_id="NSW1",
                    bands=(BidBand(price=30.0, quantity=100.0),),
                    max_capacity_mw=100.0,
                ),
                Generator(
                    unit_id="G1",
                    region_id="NSW1",
                    bands=(BidBand(price=40.0, quantity=100.0),),
                    max_capacity_mw=100.0,
                ),
            ),
            interconnectors=(),
        )


def test_valid_minimal_snapshot_constructs():
    DispatchSnapshot(
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
