
"""
Synthetic vehicle-ferry environment + baseline Full-MIP solver.

Purpose
-------
Creates reproducible synthetic ferry-loading instances and solves a static
single-epoch full mixed-integer model. This is the first computational layer
for the NGFLO manuscript. The generator is synthetic and is NOT calibrated to
a specific operator or vessel.

Model currently enforces:
- exactly one outcome per vehicle: load into one compatible lane, or defer;
- vehicle/lane width and height compatibility;
- lane residual length capacity;
- lane mass capacity;
- deck total mass capacity;
- a linear transverse-balance envelope;
- optional deck-class restrictions for heavy vehicle classes.

The baseline intentionally does NOT yet enforce exact within-lane sequencing,
ramp-conflict logic, ballast/hydrostatics, axle-load constraints, or fire-zone
rules. Those are reserved for later operator-specific extensions.

Dependencies: numpy, pandas, scipy>=1.9
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import math
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


@dataclass
class FerryConfig:
    name: str
    decks: int
    lanes_per_deck: int
    lane_length: float
    lane_width: float
    lane_height_by_deck: Tuple[float, ...]
    lane_mass_capacity: float
    deck_mass_capacity: float
    transverse_balance_ratio: float = 0.16

    @property
    def n_positions(self) -> int:
        return self.decks * self.lanes_per_deck


VEHICLE_CLASSES: Dict[str, Dict[str, float]] = {
    "car":        {"p": 0.40, "L": 4.5,  "W": 1.82, "H": 1.55, "M": 1.55},
    "suv":        {"p": 0.18, "L": 4.9,  "W": 1.92, "H": 1.78, "M": 2.10},
    "van":        {"p": 0.14, "L": 5.6,  "W": 2.05, "H": 2.45, "M": 3.20},
    "rigid_truck":{"p": 0.10, "L": 9.0,  "W": 2.45, "H": 3.60, "M": 12.0},
    "coach":      {"p": 0.06, "L": 12.0, "W": 2.50, "H": 3.70, "M": 17.0},
    "artic":      {"p": 0.12, "L": 16.2, "W": 2.50, "H": 4.00, "M": 29.0},
}

FERRY_PRESETS = {
    "small": FerryConfig(
        "small", decks=2, lanes_per_deck=8, lane_length=62.0,
        lane_width=2.75, lane_height_by_deck=(4.50, 2.60),
        lane_mass_capacity=78.0, deck_mass_capacity=510.0,
        transverse_balance_ratio=0.18
    ),
    "medium": FerryConfig(
        "medium", decks=3, lanes_per_deck=12, lane_length=78.0,
        lane_width=2.80, lane_height_by_deck=(4.60, 4.20, 2.70),
        lane_mass_capacity=95.0, deck_mass_capacity=900.0,
        transverse_balance_ratio=0.16
    ),
    "large": FerryConfig(
        "large", decks=4, lanes_per_deck=18, lane_length=95.0,
        lane_width=2.85, lane_height_by_deck=(4.70, 4.40, 3.20, 2.70),
        lane_mass_capacity=112.0, deck_mass_capacity=1550.0,
        transverse_balance_ratio=0.15
    ),
}


def make_positions(cfg: FerryConfig) -> pd.DataFrame:
    rows = []
    # Symmetric transverse coordinates in [-1,1].
    ys = np.linspace(-1.0, 1.0, cfg.lanes_per_deck)
    for r in range(cfg.decks):
        for j in range(cfg.lanes_per_deck):
            rows.append({
                "pos_id": r * cfg.lanes_per_deck + j,
                "deck": r,
                "lane": j,
                "length_cap": cfg.lane_length,
                "width_cap": cfg.lane_width,
                "height_cap": cfg.lane_height_by_deck[r],
                "mass_cap": cfg.lane_mass_capacity,
                "y_coord": float(ys[j]),
            })
    return pd.DataFrame(rows)


def generate_vehicles(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(VEHICLE_CLASSES)
    probs = np.array([VEHICLE_CLASSES[k]["p"] for k in names], dtype=float)
    probs /= probs.sum()
    cls = rng.choice(names, size=n, p=probs)

    rows = []
    for i, c in enumerate(cls):
        base = VEHICLE_CLASSES[c]
        # Small bounded heterogeneity around class nominal values.
        length = max(3.0, rng.normal(base["L"], 0.035 * base["L"]))
        width  = max(1.5, rng.normal(base["W"], 0.02 * base["W"]))
        height = max(1.3, rng.normal(base["H"], 0.025 * base["H"]))
        mass   = max(0.8, rng.normal(base["M"], 0.06 * base["M"]))
        destination = int(rng.integers(1, 4))
        priority = int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
        rows.append({
            "veh_id": i,
            "class": c,
            "length": float(length),
            "width": float(width),
            "height": float(height),
            "mass": float(mass),
            "destination": destination,
            "priority": priority,
        })
    return pd.DataFrame(rows)


def is_compatible(v, p, cfg: FerryConfig) -> bool:
    if v["width"] > p["width_cap"] + 1e-9:
        return False
    if v["height"] > p["height_cap"] + 1e-9:
        return False
    if v["mass"] > p["mass_cap"] + 1e-9:
        return False
    # Heavy vehicles are restricted away from the top passenger-oriented deck.
    if v["class"] in {"rigid_truck", "coach", "artic"}:
        if int(p["deck"]) == cfg.decks - 1:
            return False
    return True


def build_model(vehicles: pd.DataFrame, positions: pd.DataFrame, cfg: FerryConfig):
    """
    Variables:
      x_(i,p) for compatible vehicle-position pairs
      y_i deferral binary for each vehicle
    """
    x_vars: List[Tuple[int, int]] = []
    x_index: Dict[Tuple[int, int], int] = {}

    for _, v in vehicles.iterrows():
        for _, p in positions.iterrows():
            if is_compatible(v, p, cfg):
                key = (int(v.veh_id), int(p.pos_id))
                x_index[key] = len(x_vars)
                x_vars.append(key)

    n_x = len(x_vars)
    y_index = {int(i): n_x + k for k, i in enumerate(vehicles.veh_id.tolist())}
    n_var = n_x + len(y_index)

    c = np.zeros(n_var, dtype=float)

    # Assignment costs are small; deferral penalties dominate.
    # This makes loading preferred whenever physically possible, while still
    # weakly favoring tight fit, balance, and destination/deck compatibility.
    veh_by_id = vehicles.set_index("veh_id")
    pos_by_id = positions.set_index("pos_id")
    for idx, (i, p_id) in enumerate(x_vars):
        v = veh_by_id.loc[i]
        p = pos_by_id.loc[p_id]
        fit = max(0.0, (p.length_cap - v.length) / p.length_cap)
        balance = abs(p.y_coord) * v.mass / max(1.0, cfg.lane_mass_capacity)
        # Mild preference: earlier destinations on lower decks.
        discharge_pref = 0.03 * abs((v.destination - 1) - min(int(p.deck), 2))
        c[idx] = 0.20 * fit + 0.12 * balance + discharge_pref

    for i, idx in y_index.items():
        v = veh_by_id.loc[i]
        # High enough to dominate any assignment cost.
        c[idx] = 100.0 + 18.0 * v.priority + 1.2 * v.length

    # Sparse linear constraints.
    row_ind, col_ind, data = [], [], []
    lb, ub = [], []
    row = 0

    # 1) exactly one outcome: assignment or deferral.
    for i in vehicles.veh_id:
        i = int(i)
        for p_id in positions.pos_id:
            key = (i, int(p_id))
            if key in x_index:
                row_ind.append(row); col_ind.append(x_index[key]); data.append(1.0)
        row_ind.append(row); col_ind.append(y_index[i]); data.append(1.0)
        lb.append(1.0); ub.append(1.0)
        row += 1

    # 2) lane length.
    for p_id in positions.pos_id:
        p_id = int(p_id)
        for i in vehicles.veh_id:
            i = int(i)
            key = (i, p_id)
            if key in x_index:
                row_ind.append(row); col_ind.append(x_index[key])
                data.append(float(veh_by_id.loc[i, "length"]))
        lb.append(-np.inf); ub.append(float(pos_by_id.loc[p_id, "length_cap"]))
        row += 1

    # 3) lane mass.
    for p_id in positions.pos_id:
        p_id = int(p_id)
        for i in vehicles.veh_id:
            i = int(i)
            key = (i, p_id)
            if key in x_index:
                row_ind.append(row); col_ind.append(x_index[key])
                data.append(float(veh_by_id.loc[i, "mass"]))
        lb.append(-np.inf); ub.append(float(pos_by_id.loc[p_id, "mass_cap"]))
        row += 1

    # 4) deck mass.
    for deck in range(cfg.decks):
        deck_positions = positions.loc[positions.deck == deck, "pos_id"].astype(int).tolist()
        for p_id in deck_positions:
            for i in vehicles.veh_id:
                i = int(i)
                key = (i, p_id)
                if key in x_index:
                    row_ind.append(row); col_ind.append(x_index[key])
                    data.append(float(veh_by_id.loc[i, "mass"]))
        lb.append(-np.inf); ub.append(cfg.deck_mass_capacity)
        row += 1

    # 5) transverse balance envelope:
    # |sum m*y*x| <= ratio * sum m*x
    # => sum m*(y-ratio)*x <= 0
    # => sum m*(-y-ratio)*x <= 0
    ratio = cfg.transverse_balance_ratio
    for sign in (1.0, -1.0):
        for idx, (i, p_id) in enumerate(x_vars):
            m = float(veh_by_id.loc[i, "mass"])
            y = float(pos_by_id.loc[p_id, "y_coord"])
            coeff = m * (sign * y - ratio)
            row_ind.append(row); col_ind.append(idx); data.append(coeff)
        lb.append(-np.inf); ub.append(0.0)
        row += 1

    A = coo_matrix((data, (row_ind, col_ind)), shape=(row, n_var)).tocsr()
    constraints = LinearConstraint(A, np.array(lb, dtype=float), np.array(ub, dtype=float))
    bounds = Bounds(np.zeros(n_var), np.ones(n_var))
    integrality = np.ones(n_var, dtype=int)

    meta = {
        "x_vars": x_vars,
        "x_index": x_index,
        "y_index": y_index,
        "A": A,
        "lb": np.array(lb, dtype=float),
        "ub": np.array(ub, dtype=float),
    }
    return c, integrality, bounds, constraints, meta


def solve_instance(n: int, seed: int, ferry: str = "medium",
                   time_limit: float = 60.0, mip_rel_gap: float = 1e-6):
    cfg = FERRY_PRESETS[ferry]
    vehicles = generate_vehicles(n, seed)
    positions = make_positions(cfg)
    c, integrality, bounds, constraints, meta = build_model(vehicles, positions, cfg)

    t0 = time.perf_counter()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": True,
        },
    )
    elapsed = time.perf_counter() - t0

    if res.x is None:
        return {
            "ferry": ferry, "n": n, "seed": seed, "success": False,
            "status": int(res.status), "message": str(res.message),
            "runtime_s": elapsed, "objective": np.nan,
            "loaded": np.nan, "deferred": np.nan,
            "length_utilization": np.nan, "mass_loaded": np.nan,
            "n_binary": len(c), "n_x": len(meta["x_vars"]),
            "max_violation": np.nan,
            "mip_gap": getattr(res, "mip_gap", np.nan),
            "mip_node_count": getattr(res, "mip_node_count", np.nan),
        }

    x = np.asarray(res.x)
    veh_by_id = vehicles.set_index("veh_id")
    pos_by_id = positions.set_index("pos_id")

    loaded_ids = []
    total_loaded_length = 0.0
    mass_loaded = 0.0
    lane_lengths = {int(p): 0.0 for p in positions.pos_id}
    lane_masses = {int(p): 0.0 for p in positions.pos_id}

    for idx, (i, p_id) in enumerate(meta["x_vars"]):
        if x[idx] > 0.5:
            loaded_ids.append(i)
            L = float(veh_by_id.loc[i, "length"])
            M = float(veh_by_id.loc[i, "mass"])
            total_loaded_length += L
            mass_loaded += M
            lane_lengths[p_id] += L
            lane_masses[p_id] += M

    loaded = len(set(loaded_ids))
    deferred = n - loaded
    total_lane_length = float(positions.length_cap.sum())
    util = total_loaded_length / total_lane_length

    # Numerical verification of every linear constraint.
    Ax = meta["A"].dot(x)
    upper_v = np.where(np.isfinite(meta["ub"]), np.maximum(Ax - meta["ub"], 0.0), 0.0)
    lower_v = np.where(np.isfinite(meta["lb"]), np.maximum(meta["lb"] - Ax, 0.0), 0.0)
    max_violation = float(max(np.max(upper_v), np.max(lower_v), 0.0))

    return {
        "ferry": ferry,
        "n": n,
        "seed": seed,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "runtime_s": elapsed,
        "objective": float(res.fun),
        "loaded": loaded,
        "deferred": deferred,
        "load_rate": loaded / n,
        "length_utilization": util,
        "mass_loaded": mass_loaded,
        "n_binary": len(c),
        "n_x": len(meta["x_vars"]),
        "max_violation": max_violation,
        "mip_gap": float(getattr(res, "mip_gap", np.nan)),
        "mip_node_count": float(getattr(res, "mip_node_count", np.nan)),
    }


def run_benchmark(out_csv: str, ferry: str = "medium",
                  sizes=(50, 100, 200, 400), seeds=(11, 22, 33),
                  time_limit: float = 60.0):
    rows = []
    for n in sizes:
        for seed in seeds:
            r = solve_instance(n=n, seed=seed, ferry=ferry, time_limit=time_limit)
            rows.append(r)
            print(
                f"{ferry:6s} n={n:4d} seed={seed:3d} "
                f"runtime={r['runtime_s']:.3f}s loaded={r.get('loaded')} "
                f"gap={r.get('mip_gap')} viol={r.get('max_violation')}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ferry", choices=sorted(FERRY_PRESETS), default="medium")
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--out", default="baseline_full_mip_results.csv")
    args = parser.parse_args()

    df = run_benchmark(
        out_csv=args.out,
        ferry=args.ferry,
        sizes=tuple(args.sizes),
        seeds=tuple(args.seeds),
        time_limit=args.time_limit,
    )
    print("\nSummary")
    print(
        df.groupby("n")
          .agg(
              runtime_mean=("runtime_s", "mean"),
              runtime_max=("runtime_s", "max"),
              load_rate_mean=("load_rate", "mean"),
              utilization_mean=("length_utilization", "mean"),
              deferred_mean=("deferred", "mean"),
              max_violation=("max_violation", "max"),
              mip_gap_max=("mip_gap", "max"),
          )
          .reset_index()
          .to_string(index=False)
    )


if __name__ == "__main__":
    main()
