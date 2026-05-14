# Project-wide constants for the NEM dispatch solver.
#
# These come from AEMO market rules / NEMDE conventions:
#   - The NEM has 5 pricing regions, each corresponding to an Australian state.
#   - Generators submit up to 10 price/quantity bid bands per dispatch interval.
#   - The Value of Lost Load (VOLL) is the cap on regional reference prices in $/MWh.
#   - The Market Floor is the negative price floor.
#
# VOLL and MARKET_FLOOR are not used in v1 (no unserved-energy slack variable yet)
# but are kept here so that addition is a localised change.

from typing import Literal

NEM_REGIONS: tuple[str, ...] = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")

# RegionId is a typing.Literal so that pydantic rejects typos like "NSW" or "Vic1"
# at construction time rather than producing nonsense LPs.
RegionId = Literal["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]

DEFAULT_NUM_BANDS = 10  # NEMDE convention; we don't enforce it (any N>=1 is allowed).

VOLL = 17_500.0          # $/MWh -- AEMO market price cap
MARKET_FLOOR = -1_000.0  # $/MWh -- AEMO market price floor
