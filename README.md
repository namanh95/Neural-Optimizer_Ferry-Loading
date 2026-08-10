# Synthetic Ferry Environment + Full-MIP Baseline

This package is the first computational layer for the manuscript
"A Neural-Guided Optimization Framework for Large-Scale Dynamic Vehicle-Ferry Loading."

## Files
- synthetic_ferry_full_mip.py: reproducible synthetic instance generator and static Full-MIP solver.
- baseline_full_mip_pilot.csv: initial five-instance pilot output.

## Current modeled constraints
1. One outcome per vehicle: assign to one compatible lane/deck or defer.
2. Width and height compatibility.
3. Lane-length capacity.
4. Lane mass capacity.
5. Deck total-mass capacity.
6. Linear transverse-balance envelope.
7. Heavy-vehicle deck restrictions.

## Not yet included
- exact within-lane vehicle ordering;
- explicit discharge-sequence blocking constraints;
- ramp conflicts;
- axle-load limits;
- ballast/hydrostatic equations;
- fire-zone / dangerous-goods rules;
- stochastic arrivals and rolling-horizon state transitions.

These must be added before the model is treated as an operator-complete ferry-loading model.

## Pilot interpretation
The initial pilot uses the medium synthetic ferry and seed 11 with a 5-second MIP limit.

The 50-vehicle case was essentially solved to optimality.
The 100- and 200-vehicle cases had very small reported MIP gaps.
The 300-vehicle case remained close to the current best bound.
The 400-vehicle case was strongly time-limited (reported MIP gap about 0.50) and therefore
must NOT be presented as an optimal Full-MIP benchmark.

All reported post-solve linear-constraint violations were 0.0 at displayed precision.

## Next validation stage
1. Add exact lane-position / discharge-order structure.
2. Introduce controlled congestion scenarios.
3. Run multiple random seeds for each size.
4. Use longer solve times for small/medium cases to establish certified optimal benchmarks.
5. Save both incumbent objective and lower bound for time-limited large cases.
6. Only after this baseline is validated, generate training labels for NGFLO.
