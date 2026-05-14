# `nem_solver` — Regional NEM Dispatch Solver

A Python approximation of AEMO's NEMDE (NEM Dispatch Engine). Given regional
demand, generator and storage bid stacks, and interconnector limits with loss
curves, it solves for least-cost dispatch and recovers regional reference
prices (RRPs).

This document explains what the package does, how the LP is formulated, and
how each module fits together. It assumes basic familiarity with linear
programming and the Australian NEM (regions, dispatch intervals, RRPs).

---

## What's in this version

- **5-region (state-level) dispatch**: NSW1, QLD1, VIC1, SA1, TAS1.
- **Multi-band generator bids** (NEMDE convention is 10 bands; any N ≥ 1 works).
- **Bidirectional storage** (batteries, pumped hydro) with separate charge and
  discharge bid stacks.
- **Directional interconnectors** with piecewise-linear convex loss curves and
  separate forward/reverse MW limits.
- **Single-period solver** (`solve_dispatch`) and a **time-series wrapper**
  (`run_timeseries`) that solves a sequence of independent snapshots.
- **5-minute and 30-minute resolution** support (any positive interval works).

### Out of scope (deliberately deferred)

- FCAS co-optimisation. The data model has a hook (`Generator.fcas_offers`)
  but the LP is energy-only.
- Storage state-of-charge linkage across periods. Each period is solved
  independently in v1.
- Intra-state nodes / network constraints. Granular extension is a future
  rewrite candidate; the data model abstracts a `Region` so the LP would not
  need rebuilding to add them.
- AEMO data ingestion. Inputs are passed in via `DispatchSnapshot` objects.
- Unserved-energy slack at VOLL.
- Ramp limits, minimum generation levels, unit commitment.

---

## File map

| File | Purpose |
|---|---|
| [constants.py](constants.py) | NEM region IDs, NEMDE bid-band convention, VOLL/floor constants |
| [schemas.py](schemas.py) | pydantic v2 input/output models with cross-field validators |
| [losses.py](losses.py) | Pure-Python PWL loss helpers (no LP dependency) |
| [model.py](model.py) | `DispatchModel` — builds the linopy LP and extracts results |
| [solver.py](solver.py) | `solve_dispatch(...)` — thin functional wrapper |
| [timeseries.py](timeseries.py) | `run_timeseries(...)` — sequence orchestration |
| [__init__.py](__init__.py) | Re-exports the public API |

Tests live at the repo root in [`tests/`](../../../tests).

---

## Quickstart

```python
from datetime import datetime
from nem_solver import (
    BidBand, DispatchSnapshot, Generator, Interconnector, PWLLossSegment,
    Region, solve_dispatch,
)

snapshot = DispatchSnapshot(
    timestamp=datetime(2026, 5, 8, 12, 0),
    regions=(
        Region(region_id="NSW1", demand_mw=100.0),
        Region(region_id="VIC1", demand_mw=100.0),
    ),
    generators=(
        Generator(
            unit_id="NSW_GEN", region_id="NSW1",
            bands=(BidBand(price=40.0, quantity=400.0),),
            max_capacity_mw=400.0,
        ),
        Generator(
            unit_id="VIC_GEN", region_id="VIC1",
            bands=(BidBand(price=120.0, quantity=400.0),),
            max_capacity_mw=400.0,
        ),
    ),
    interconnectors=(
        Interconnector(
            ic_id="NSW1-VIC1", from_region="NSW1", to_region="VIC1",
            forward_limit_mw=200.0, reverse_limit_mw=200.0,
            forward_loss_segments=(
                PWLLossSegment(flow_from_mw=0.0, flow_to_mw=200.0,
                               marginal_loss_factor=0.0),
            ),
            reverse_loss_segments=(
                PWLLossSegment(flow_from_mw=0.0, flow_to_mw=200.0,
                               marginal_loss_factor=0.0),
            ),
        ),
    ),
)

result = solve_dispatch(snapshot, resolution_minutes=5)
print(result.rrp_by_region)         # {'NSW1': 40.0, 'VIC1': 40.0}
print(result.interconnector_flows)  # NSW exports 100 MW into VIC
```

For a sequence of snapshots:

```python
from nem_solver import run_timeseries

ts = run_timeseries([snap_t0, snap_t1, ...], resolution_minutes=5)
ts.prices    # wide-form: index timestamp, columns region IDs
ts.dispatch  # long-form: (timestamp, unit_id, dispatched_mw)
ts.flows     # long-form: (timestamp, ic_id, flow_mw, losses_mw)
```

---

## Data model (`schemas.py`)

All inputs and outputs are pydantic v2 `BaseModel`s with `frozen=True` and
`extra="forbid"`. Cross-field invariants are enforced via `@model_validator`
hooks at construction time, so a malformed snapshot fails before it reaches
the LP builder.

### Inputs

| Type | Fields | Key invariants |
|---|---|---|
| `BidBand` | `price`, `quantity` | quantity ≥ 0 |
| `Region` | `region_id`, `demand_mw` | region_id is one of the 5 NEM regions |
| `Generator` | `unit_id`, `region_id`, `bands`, `max_capacity_mw`, `min_capacity_mw`, `fcas_offers` | bid prices non-decreasing; band quantities sum to capacity |
| `BidirectionalUnit` | `unit_id`, `region_id`, `discharge_bands`, `charge_bands`, `max_discharge_mw`, `max_charge_mw`, `round_trip_efficiency` | discharge prices non-decreasing; charge prices **non-increasing** (see below); quantities aligned |
| `PWLLossSegment` | `flow_from_mw`, `flow_to_mw`, `marginal_loss_factor` | width > 0; MLF ≥ 0 |
| `Interconnector` | `ic_id`, `from_region`, `to_region`, `forward_limit_mw`, `reverse_limit_mw`, `forward_loss_segments`, `reverse_loss_segments` | endpoints distinct; segments contiguous from 0 with non-decreasing MLFs and total coverage ≥ directional limit |
| `DispatchSnapshot` | `timestamp`, `regions`, `generators`, `bidirectional_units`, `interconnectors` | all unit/IC region references resolve; unit and IC IDs unique |

### Outputs

| Type | Description |
|---|---|
| `UnitDispatch` | One per generator: total dispatched MW plus per-band breakdown |
| `StorageDispatch` | One per bidirectional unit: discharge_mw, charge_mw, net_mw, plus per-band breakdowns |
| `InterconnectorFlow` | Signed flow (+ = forward), losses, sent_out, received |
| `DispatchResult` | timestamp + objective value + `rrp_by_region` + the lists above + solver_status |

### Charge band ordering convention

Discharge bands work like generator bands: cheapest band fills first, prices
non-decreasing across the stack.

Charge bands are **non-increasing**: the band with the highest willingness-to-pay
goes first. In the LP, charge variables get a coefficient of `-price`, so the
band with the largest price has the most negative coefficient and the LP fills
it first. Storing them in descending order keeps "band index 0 = most desirable
to fill", which mirrors the generator/discharge convention.

---

## LP formulation (`model.py`)

The dispatch LP is a single-period linear program with continuous decision
variables, linear constraints, and a linear cost objective. No binaries are
needed; see the section [No-binary tricks](#no-binary-tricks) for why.

### Sets

- $\mathcal{R}$ — regions (NSW1, QLD1, VIC1, SA1, TAS1)
- $\mathcal{G}$ — generators; $\mathcal{G}_r \subseteq \mathcal{G}$ are those in region $r$
- $\mathcal{S}$ — bidirectional storage units; $\mathcal{S}_r$ those in region $r$
- $\mathcal{K}$ — interconnectors; for $k \in \mathcal{K}$ let $f(k)$ and $t(k)$ denote its from- and to-region
- $\mathcal{B}_g$ — bid bands of generator $g$; $\mathcal{B}_s^{\text{dis}}, \mathcal{B}_s^{\text{chg}}$ — discharge / charge bands of storage $s$
- $\mathcal{L}_k^{\text{fwd}}, \mathcal{L}_k^{\text{rev}}$ — PWL loss segments of interconnector $k$ in each direction

### Parameters

- $\pi_{g,b}^{\text{gen}}, \pi_{s,b}^{\text{dis}}, \pi_{s,b}^{\text{chg}}$ — bid prices ($/MWh)
- $\bar{Q}_{g,b}^{\text{gen}}, \bar{Q}_{s,b}^{\text{dis}}, \bar{Q}_{s,b}^{\text{chg}}$ — band quantities (MW)
- $\bar{F}_k^{\text{fwd}}, \bar{F}_k^{\text{rev}}$ — directional MW limits of interconnector $k$
- $w_{k,l}^{\text{fwd}}, w_{k,l}^{\text{rev}}$ — width of PWL segment $l$ on interconnector $k$ (MW)
- $\mu_{k,l}^{\text{fwd}}, \mu_{k,l}^{\text{rev}}$ — marginal loss factor on segment $l$ (dimensionless, non-decreasing in $l$)
- $D_r$ — operational demand in region $r$ (MW)
- $\Delta t$ — interval length in hours ($\tfrac{5}{60}$ for 5-min, $\tfrac{1}{2}$ for 30-min)

### Decision variables

All variables are continuous and non-negative.

| Variable | Implementation | Index | Upper bound | Meaning |
|---|---|---|---|---|
| $x_{g,b}^{\text{gen}}$ | `gen_band` | $(g,b)$ | $\bar{Q}_{g,b}^{\text{gen}}$ | MW dispatched in band $b$ of generator $g$ |
| $x_{s,b}^{\text{dis}}$ | `dis_band` | $(s,b)$ | $\bar{Q}_{s,b}^{\text{dis}}$ | MW discharged in band $b$ of storage $s$ |
| $x_{s,b}^{\text{chg}}$ | `chg_band` | $(s,b)$ | $\bar{Q}_{s,b}^{\text{chg}}$ | MW charged in band $b$ of storage $s$ |
| $f_k^{\text{fwd}}$ | `flow_fwd` | $k$ | $\bar{F}_k^{\text{fwd}}$ | MW from $f(k)$ to $t(k)$ on interconnector $k$ |
| $f_k^{\text{rev}}$ | `flow_rev` | $k$ | $\bar{F}_k^{\text{rev}}$ | MW from $t(k)$ to $f(k)$ on interconnector $k$ |
| $\sigma_{k,l}^{\text{fwd}}$ | `seg_fwd` | $(k,l)$ | $w_{k,l}^{\text{fwd}}$ | MW assigned to forward PWL segment $l$ |
| $\sigma_{k,l}^{\text{rev}}$ | `seg_rev` | $(k,l)$ | $w_{k,l}^{\text{rev}}$ | MW assigned to reverse PWL segment $l$ |

### Constraints

**(1) PWL segment partition** — for each interconnector $k$ and direction, the directional flow equals the sum of its segment activations:

$$\sum_{l \in \mathcal{L}_k^{\text{fwd}}} \sigma_{k,l}^{\text{fwd}} = f_k^{\text{fwd}}, \qquad \sum_{l \in \mathcal{L}_k^{\text{rev}}} \sigma_{k,l}^{\text{rev}} = f_k^{\text{rev}} \qquad \forall k \in \mathcal{K}$$

Directional losses are then linear expressions over the segment variables (not separate decision variables):

$$L_k^{\text{fwd}} = \sum_{l \in \mathcal{L}_k^{\text{fwd}}} \mu_{k,l}^{\text{fwd}} \, \sigma_{k,l}^{\text{fwd}}, \qquad L_k^{\text{rev}} = \sum_{l \in \mathcal{L}_k^{\text{rev}}} \mu_{k,l}^{\text{rev}} \, \sigma_{k,l}^{\text{rev}}$$

**(2) Regional energy balance** — one equality per region. This is the constraint named `energy_balance` whose dual yields the regional reference price. For each $r \in \mathcal{R}$:

$$
\begin{aligned}
& \sum_{g \in \mathcal{G}_r} \sum_{b \in \mathcal{B}_g} x_{g,b}^{\text{gen}} && \text{(local generation)} \\
&+ \sum_{s \in \mathcal{S}_r} \sum_{b \in \mathcal{B}_s^{\text{dis}}} x_{s,b}^{\text{dis}} && \text{(local discharge)} \\
&- \sum_{s \in \mathcal{S}_r} \sum_{b \in \mathcal{B}_s^{\text{chg}}} x_{s,b}^{\text{chg}} && \text{(local charge)} \\
&+ \sum_{\{k \,:\, t(k) = r\}} \left( f_k^{\text{fwd}} - L_k^{\text{fwd}} \right) && \text{(forward inflow, net of losses)} \\
&+ \sum_{\{k \,:\, f(k) = r\}} \left( f_k^{\text{rev}} - L_k^{\text{rev}} \right) && \text{(reverse inflow, net of losses)} \\
&- \sum_{\{k \,:\, f(k) = r\}} f_k^{\text{fwd}} && \text{(forward outflow, gross)} \\
&- \sum_{\{k \,:\, t(k) = r\}} f_k^{\text{rev}} && \text{(reverse outflow, gross)} \\
&\quad = D_r
\end{aligned}
$$

Sender regions account for the gross sent-out MW; receiver regions account for the post-loss MW. This means the loss MW are charged to the receiver, which matches the AEMO settlement convention.

**(3) Variable bounds** — interconnector and band capacity limits are encoded as upper bounds on the decision variables (item-by-item in the table above) rather than as explicit constraints. The LP solver handles them natively.

### Objective

Minimise total dispatch cost over the interval:

$$
\min \; \Delta t \left[ \sum_{g \in \mathcal{G}} \sum_{b \in \mathcal{B}_g} \pi_{g,b}^{\text{gen}} \, x_{g,b}^{\text{gen}} \;+\; \sum_{s \in \mathcal{S}} \sum_{b \in \mathcal{B}_s^{\text{dis}}} \pi_{s,b}^{\text{dis}} \, x_{s,b}^{\text{dis}} \;-\; \sum_{s \in \mathcal{S}} \sum_{b \in \mathcal{B}_s^{\text{chg}}} \pi_{s,b}^{\text{chg}} \, x_{s,b}^{\text{chg}} \right]
$$

The negative sign on charge variables represents the storage unit's willingness to pay to consume: charging at $\pi_{s,b}^{\text{chg}}$ \$/MWh reduces the objective by $\pi_{s,b}^{\text{chg}} \cdot x_{s,b}^{\text{chg}} \cdot \Delta t$ dollars, so the LP charges whenever the regional price falls below the charge bid — exactly the right economic behaviour.

The leading $\Delta t$ converts a power-cost rate (\$/MWh × MW) into actual dollars over the interval, making the objective and its duals dimensionally meaningful.

### Recovering regional reference prices (RRPs)

Let $\lambda_r$ denote the dual of the energy balance constraint for region $r$. By LP duality, $\lambda_r$ equals the marginal cost of supplying one additional unit of demand in region $r$ — exactly the RRP, up to scaling. Because the objective was scaled by $\Delta t$, the raw dual is in dollars per interval; dividing recovers \$/MWh:

$$
\text{RRP}_r \;=\; \frac{\lambda_r}{\Delta t}
$$

In code:

```python
duals = model.constraints["energy_balance"].dual         # xarray, indexed by region
rrp = duals.sel(region="NSW1").item() / resolution_hours # $/MWh
```

---

## No-binary tricks

The LP would naively call for two integer/binary structures. Both are
unnecessary in this formulation:

### 1. PWL segment ordering

Naïve: "fill segment 0 fully before opening segment 1" needs SOS2 or binaries.

Why we don't need them: the schema requires marginal loss factors to be
**non-decreasing** across segments (a convex loss curve). When the LP
minimises cost, every MW of flow assigned to a high-MLF segment is more
expensive (in losses) than assigning it to a low-MLF segment. So the LP
optimum always fills the lowest-MLF segment first, then the next, and so on
— without any explicit ordering constraint.

This trick **breaks** if you ever feed in a non-convex loss curve (MLFs
decreasing somewhere). The validator catches this at construction time.

### 2. Storage non-simultaneous charge/discharge

Naïve: forbid `chg > 0 ∧ dis > 0` with a binary.

Why we don't need it: with a sane bid stack, charge prices are below
discharge prices. Any candidate solution with both `chg > 0` and `dis > 0`
can be reduced — netting them lowers the objective — so the LP optimum has
`chg * dis == 0` automatically. We assert this in tests rather than
enforce it.

This trick **breaks** for pathological bid stacks where the highest charge
price exceeds the lowest discharge price. If the test
`test_no_simultaneous_charge_discharge` ever fails, that's the signal to add
a binary-free reformulation (e.g. bid-stack alignment).

---

## Linopy patterns used

A short cheat sheet for reading `model.py`:

- **Flat 1-D variables.** Each "block" (gen bands, storage discharge bands, IC
  forward flows, etc.) is a single 1-D linopy `Variable` indexed 0..N-1.
  Parallel Python lists (`_BandIndex`, `_SegmentIndex`) record which unit /
  region / segment each entry belongs to.
- **Group sums via `.isel(...)`.** Per-region supply terms are built by
  collecting the integer positions of variables that belong to the region
  and slicing with `.isel(dim=indices).sum()`.
- **Region-indexed constraints via `linopy.expressions.merge`.** Building a
  list of per-region scalar linexprs and calling
  `merge(exprs, dim="region").assign_coords(region=...)` produces a single
  multi-row constraint whose dual is one xarray DataArray indexed by region.
  Much cleaner than five separately-named constraints.
- **HiGHS as solver.** Installed as `highspy` and selected via
  `model.solve(solver_name="highs")`. linopy returns LP duals on equality
  constraints directly via `constraint.dual`.

---

## Time-series orchestration (`timeseries.py`)

`run_timeseries(snapshots)` iterates each snapshot, calls `solve_dispatch`,
and aggregates the results into four DataFrames:

- `prices` — wide-form, indexed by timestamp, one column per region
- `dispatch` — long-form `(timestamp, unit_id, dispatched_mw)`
- `storage_dispatch` — long-form with `discharge_mw`, `charge_mw`, `net_mw`
- `flows` — long-form `(timestamp, ic_id, flow_mw, losses_mw)`

Plus `raw: list[DispatchResult]` for diagnostics.

Each period is solved independently. Future improvements (when needed):

- **Storage SOC linkage.** Add a `prev_soc_mwh` argument and a per-storage
  SOC update step between periods, plus a constraint that `chg_mw` and
  `dis_mw` respect the available headroom in this period.
- **Parallel solves.** Add an `n_jobs` parameter and switch the loop to
  `joblib.Parallel(...)`. Each period is independent so this is trivial.
- **Warm starts.** linopy + HiGHS can warm-start, which would matter for
  hour/day forecasts where consecutive periods are similar. The
  `DispatchModel` class is the seam where this would live.

---

## Testing

The test suite (28 tests, all passing) lives at the repo root in `tests/`:

| File | Coverage |
|---|---|
| `test_schemas.py` | Validator behaviour: bad band orders, region typos, IC topology, etc. |
| `test_losses.py` | PWL math helpers in isolation |
| `test_single_region.py` | One-region, multi-band stacking, RRP equals marginal band price |
| `test_two_region.py` | Unconstrained and constrained interconnector behaviour |
| `test_pwl_losses.py` | LP loss recovery matches analytic PWL evaluation; segment fill order |
| `test_battery.py` | Discharge at high prices, charge at low prices, no simultaneous |
| `test_timeseries.py` | 5-min and 30-min resolutions; price response to demand |

Run with:

```bash
uv run pytest -v
```

### LP degeneracy gotcha

Demands that fall exactly on a band boundary (e.g. demand = 100 MW with bands
of 50 MW each) produce a degenerate LP where the dual could be the price of
the band ending at that boundary or the price of the next band. HiGHS picks
one based on its pivot path, and this is mathematically correct — RRP at a
boundary is genuinely set-valued. Tests use non-corner demands to avoid this.

---

## Design decisions worth understanding

**Pydantic over dataclasses.** Inputs will eventually come from heterogeneous
sources (AEMO MMS, CSVs, parquet). Runtime validation matters more than the
microsecond cost of `BaseModel` construction. Cross-field validators
(monotonic prices, MLF convexity, region topology) catch malformed inputs
clearly rather than producing nonsense LPs.

**Frozen models.** Snapshots are hashable and safe to share across solves;
results are immutable, which makes them safe to keep in a `raw` list of
historical results without worrying about aliasing.

**Flat 1-D variables.** Generators have different numbers of bands so a
proper 2-D `(unit, band)` xarray is awkward. A flat dim with parallel
metadata lists is simpler to build, simpler to slice for region balance,
and gives the same LP.

**Class + functional wrapper.** `DispatchModel` is the reusable object
(useful when adding warm-starts or coefficient updates later);
`solve_dispatch` is the ergonomic entry point. Most callers should use the
function form.

**Pandas (not polars).** linopy is built on xarray, which interoperates with
pandas natively. Polars would mean conversions at every boundary for no
gain at this scale.

---

## Extending: typical questions

- **Add another region (intra-state node).** Add it to `NEM_REGIONS` and
  `RegionId` in `constants.py`. Existing data model and LP code will accept
  it without further changes; you'll need new interconnector definitions
  for the topology.
- **Add FCAS.** Reuse `Generator.fcas_offers`. Add per-service FCAS
  variables in `model.py`, FCAS requirement constraints (one per region per
  service), and energy/FCAS capacity coupling constraints. The objective
  picks up FCAS bid prices.
- **Add storage SOC.** Extend `BidirectionalUnit` with `initial_soc_mwh`
  and `capacity_mwh`; in the time-series wrapper, pass post-period SOC into
  the next snapshot, and bound charge/discharge by the available SOC
  headroom in the LP.
- **Replace MW limits with constraint equations.** AEMO uses generic
  constraint equations (LHS coefficients on units, RHS values). Add a
  `Constraint` schema and a method on `DispatchModel` that translates each
  constraint into a linopy `add_constraints` call.
- **Change the solver.** Pass `solver_name="gurobi"` or `"cbc"` to
  `solve_dispatch`. linopy supports several backends; HiGHS is the default
  because it's open source and bundled.
