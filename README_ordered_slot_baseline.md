# Ordered-Slot Full-MIP Baseline

This is the second computational baseline for the manuscript
"A Neural-Guided Optimization Framework for Large-Scale Dynamic Vehicle-Ferry Loading."

## New features relative to the first prototype
- Explicit ordered slots inside every lane.
- At most one vehicle per slot.
- No-gap occupancy: deeper slots cannot be occupied before shallower slots.
- Exact monotone discharge order along each occupied lane.
- Controlled fleet-composition regimes: balanced, heavy, and car-dense.

## Slot convention
Slot 0 is closest to the discharge ramp. Destination rank 1 means earlier discharge.
The model enforces nondecreasing destination rank with increasing slot depth.
Therefore an earlier-discharge vehicle cannot be blocked by a later-discharge vehicle
within the same lane under this simplified single-ramp lane representation.

## Pilot validation
Eight small-ferry cases were run with a 5-second MIP time limit:
- n = 30 and 50 vehicles;
- balanced and heavy fleet mixtures;
- seeds 11 and 22.

Observed:
- discharge-order inversions: 0 in every case;
- maximum linear-constraint residuals were numerical roundoff only (about 1e-12 or smaller);
- n=30 cases solved to certified zero MIP gap;
- balanced n=50 cases remained near optimal within 5 seconds;
- one heavy n=50 case had a very large reported MIP gap (~0.94), so its incumbent must not
  be described as optimal.

## Important limitation
The ordered-slot model is much stronger than the first baseline but is still synthetic.
It does not yet include:
- exact ramp-network conflicts;
- axle-load constraints;
- ballast and hydrostatic stability calculations;
- dangerous-goods/fire-zone rules;
- operator-specific deck geometry;
- stochastic arrivals and rolling-horizon re-optimization.

## Next computational stage
1. Add benchmark certification with longer solve times for small instances.
2. Run multiple seeds across congestion regimes.
3. Introduce dynamic arrivals and rolling-horizon re-optimization.
4. Save optimal or near-optimal solutions as labels for NGFLO only after benchmark certification.
