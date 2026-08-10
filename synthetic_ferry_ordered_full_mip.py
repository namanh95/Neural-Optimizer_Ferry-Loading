
"""
Synthetic ferry environment + ordered-slot Full-MIP benchmark.

Adds to the first baseline:
- explicit ordered positions within every lane;
- one vehicle per ordered slot;
- no-gap occupancy within a lane;
- exact discharge-order monotonicity along each occupied lane;
- controlled congestion scenarios.

Interpretation of slot order
----------------------------
Slot k=0 is closest to the discharge ramp. Smaller destination rank means
earlier discharge. Therefore, destination ranks must be nondecreasing as
vehicles are placed deeper into a lane.

This is still a synthetic benchmark and is NOT an operator-complete vessel
model. Ballast/hydrostatics, axle loads, dangerous-goods rules, and detailed
ramp-conflict constraints remain outside the current baseline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
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
    max_slots_per_lane: int
    transverse_balance_ratio: float = 0.16

    @property
    def n_positions(self):
        return self.decks * self.lanes_per_deck


VEHICLE_CLASSES = {
    "car":         {"p": 0.40, "L": 4.5,  "W": 1.82, "H": 1.55, "M": 1.55},
    "suv":         {"p": 0.18, "L": 4.9,  "W": 1.92, "H": 1.78, "M": 2.10},
    "van":         {"p": 0.14, "L": 5.6,  "W": 2.05, "H": 2.45, "M": 3.20},
    "rigid_truck": {"p": 0.10, "L": 9.0,  "W": 2.45, "H": 3.60, "M": 12.0},
    "coach":       {"p": 0.06, "L": 12.0, "W": 2.50, "H": 3.70, "M": 17.0},
    "artic":       {"p": 0.12, "L": 16.2, "W": 2.50, "H": 4.00, "M": 29.0},
}

FERRY_PRESETS = {
    "small": FerryConfig(
        "small", 2, 8, 62.0, 2.75, (4.50, 2.60),
        78.0, 510.0, max_slots_per_lane=14, transverse_balance_ratio=0.18
    ),
    "medium": FerryConfig(
        "medium", 3, 12, 78.0, 2.80, (4.60, 4.20, 2.70),
        95.0, 900.0, max_slots_per_lane=18, transverse_balance_ratio=0.16
    ),
    "large": FerryConfig(
        "large", 4, 18, 95.0, 2.85, (4.70, 4.40, 3.20, 2.70),
        112.0, 1550.0, max_slots_per_lane=20, transverse_balance_ratio=0.15
    ),
}


def make_positions(cfg: FerryConfig) -> pd.DataFrame:
    rows = []
    ys = np.linspace(-1.0, 1.0, cfg.lanes_per_deck)
    for r in range(cfg.decks):
        for j in range(cfg.lanes_per_deck):
            rows.append({
                "pos_id": r * cfg.lanes_per_deck + j,
                "deck": r, "lane": j,
                "length_cap": cfg.lane_length,
                "width_cap": cfg.lane_width,
                "height_cap": cfg.lane_height_by_deck[r],
                "mass_cap": cfg.lane_mass_capacity,
                "y_coord": float(ys[j]),
            })
    return pd.DataFrame(rows)


def generate_vehicles(n: int, seed: int, congestion: str = "balanced") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(VEHICLE_CLASSES)
    probs = np.array([VEHICLE_CLASSES[k]["p"] for k in names], float)

    if congestion == "heavy":
        probs = np.array([0.25, 0.12, 0.13, 0.16, 0.09, 0.25], float)
    elif congestion == "car_dense":
        probs = np.array([0.58, 0.20, 0.10, 0.05, 0.02, 0.05], float)
    elif congestion != "balanced":
        raise ValueError("congestion must be balanced, heavy, or car_dense")
    probs /= probs.sum()

    cls = rng.choice(names, size=n, p=probs)
    rows = []
    for i, c in enumerate(cls):
        b = VEHICLE_CLASSES[c]
        length = max(3.0, rng.normal(b["L"], 0.035*b["L"]))
        width  = max(1.5, rng.normal(b["W"], 0.020*b["W"]))
        height = max(1.3, rng.normal(b["H"], 0.025*b["H"]))
        mass   = max(0.8, rng.normal(b["M"], 0.060*b["M"]))
        rows.append({
            "veh_id": i,
            "class": c,
            "length": float(length),
            "width": float(width),
            "height": float(height),
            "mass": float(mass),
            "destination": int(rng.integers(1, 4)),
            "priority": int(rng.choice([1,2,3], p=[0.60,0.30,0.10])),
        })
    return pd.DataFrame(rows)


def compatible(v, p, cfg):
    if v.width > p.width_cap + 1e-9 or v.height > p.height_cap + 1e-9:
        return False
    if v.mass > p.mass_cap + 1e-9:
        return False
    if v["class"] in {"rigid_truck", "coach", "artic"} and int(p.deck) == cfg.decks - 1:
        return False
    return True


def build_ordered_slot_model(vehicles, positions, cfg):
    veh = vehicles.set_index("veh_id")
    pos = positions.set_index("pos_id")

    # x(i,p,k): vehicle i is put in slot k of lane/deck position p.
    x_vars = []
    x_idx = {}
    for _, v in vehicles.iterrows():
        for _, p in positions.iterrows():
            if compatible(v, p, cfg):
                for k in range(cfg.max_slots_per_lane):
                    key = (int(v.veh_id), int(p.pos_id), k)
                    x_idx[key] = len(x_vars)
                    x_vars.append(key)

    n_x = len(x_vars)
    y_idx = {int(i): n_x + q for q, i in enumerate(vehicles.veh_id)}
    nvar = n_x + len(y_idx)

    c = np.zeros(nvar)
    for q, (i, p_id, k) in enumerate(x_vars):
        v = veh.loc[i]
        p = pos.loc[p_id]
        fit = max(0.0, (p.length_cap - v.length) / p.length_cap)
        balance = abs(p.y_coord) * v.mass / max(1.0, cfg.lane_mass_capacity)
        # Weak preference for earlier discharge closer to ramp.
        sequencing = 0.008 * k * (4 - v.destination)
        c[q] = 0.16*fit + 0.10*balance + sequencing

    for i, q in y_idx.items():
        v = veh.loc[i]
        c[q] = 100.0 + 18.0*v.priority + 1.2*v.length

    ri, ci, da = [], [], []
    lb, ub = [], []
    row = 0

    def add(coeffs, lo, hi):
        nonlocal row
        for col, val in coeffs:
            if val != 0:
                ri.append(row); ci.append(col); da.append(float(val))
        lb.append(lo); ub.append(hi)
        row += 1

    # 1) Each vehicle assigned once or deferred.
    for i in vehicles.veh_id.astype(int):
        coeffs = [(idx, 1.0) for key, idx in x_idx.items() if key[0] == i]
        coeffs.append((y_idx[i], 1.0))
        add(coeffs, 1.0, 1.0)

    # 2) At most one vehicle per ordered slot.
    for p_id in positions.pos_id.astype(int):
        for k in range(cfg.max_slots_per_lane):
            coeffs = [(idx, 1.0) for (i,p,kk), idx in x_idx.items() if p == p_id and kk == k]
            add(coeffs, -np.inf, 1.0)

    # 3) No gaps: occupancy(k+1) <= occupancy(k).
    for p_id in positions.pos_id.astype(int):
        for k in range(cfg.max_slots_per_lane - 1):
            coeffs = []
            for (i,p,kk), idx in x_idx.items():
                if p == p_id and kk == k+1:
                    coeffs.append((idx, 1.0))
                elif p == p_id and kk == k:
                    coeffs.append((idx, -1.0))
            add(coeffs, -np.inf, 0.0)

    # 4) Lane length.
    for p_id in positions.pos_id.astype(int):
        coeffs = []
        for (i,p,k), idx in x_idx.items():
            if p == p_id:
                coeffs.append((idx, veh.loc[i, "length"]))
        add(coeffs, -np.inf, pos.loc[p_id, "length_cap"])

    # 5) Lane mass.
    for p_id in positions.pos_id.astype(int):
        coeffs = []
        for (i,p,k), idx in x_idx.items():
            if p == p_id:
                coeffs.append((idx, veh.loc[i, "mass"]))
        add(coeffs, -np.inf, pos.loc[p_id, "mass_cap"])

    # 6) Deck mass.
    for deck in range(cfg.decks):
        pset = set(positions.loc[positions.deck == deck, "pos_id"].astype(int))
        coeffs = []
        for (i,p,k), idx in x_idx.items():
            if p in pset:
                coeffs.append((idx, veh.loc[i, "mass"]))
        add(coeffs, -np.inf, cfg.deck_mass_capacity)

    # 7) Exact discharge-order monotonicity.
    # Let D_k = sum_i d_i x_i,p,k and O_k = sum_i x_i,p,k.
    # Enforce D_k <= D_(k+1) + Dmax*(1 - O_(k+1)).
    # Because no gaps are allowed, if k+1 is occupied then k is occupied.
    Dmax = int(vehicles.destination.max())
    for p_id in positions.pos_id.astype(int):
        for k in range(cfg.max_slots_per_lane - 1):
            coeffs = []
            for (i,p,kk), idx in x_idx.items():
                if p != p_id:
                    continue
                if kk == k:
                    coeffs.append((idx, veh.loc[i, "destination"]))
                elif kk == k+1:
                    coeffs.append((idx, -veh.loc[i, "destination"] + Dmax))
            # D_k - D_(k+1) + Dmax*O_(k+1) <= Dmax
            add(coeffs, -np.inf, float(Dmax))

    # 8) Transverse balance.
    ratio = cfg.transverse_balance_ratio
    for sign in (1.0, -1.0):
        coeffs = []
        for (i,p_id,k), idx in x_idx.items():
            coeffs.append(
                (idx, veh.loc[i, "mass"] * (sign*pos.loc[p_id, "y_coord"] - ratio))
            )
        add(coeffs, -np.inf, 0.0)

    A = coo_matrix((da, (ri, ci)), shape=(row, nvar)).tocsr()
    return (
        c,
        np.ones(nvar, dtype=int),
        Bounds(np.zeros(nvar), np.ones(nvar)),
        LinearConstraint(A, np.array(lb), np.array(ub)),
        {"x_vars": x_vars, "x_idx": x_idx, "y_idx": y_idx, "A": A,
         "lb": np.array(lb), "ub": np.array(ub)}
    )


def solve(n, seed, ferry="small", congestion="balanced", time_limit=10.0, gap=1e-6):
    cfg = FERRY_PRESETS[ferry]
    vehicles = generate_vehicles(n, seed, congestion)
    positions = make_positions(cfg)
    c, integrality, bounds, constraints, meta = build_ordered_slot_model(vehicles, positions, cfg)

    t0 = time.perf_counter()
    res = milp(
        c=c, integrality=integrality, bounds=bounds, constraints=constraints,
        options={"time_limit": float(time_limit), "mip_rel_gap": float(gap), "presolve": True}
    )
    elapsed = time.perf_counter() - t0

    base = {
        "ferry": ferry, "congestion": congestion, "n": n, "seed": seed,
        "runtime_s": elapsed, "status": int(res.status), "message": str(res.message),
        "n_binary": len(c), "n_x": len(meta["x_vars"]),
        "success": bool(res.x is not None),
        "mip_gap": float(getattr(res, "mip_gap", np.nan)),
        "mip_node_count": float(getattr(res, "mip_node_count", np.nan)),
    }
    if res.x is None:
        base.update({"objective": np.nan, "loaded": np.nan, "deferred": np.nan,
                     "load_rate": np.nan, "length_utilization": np.nan,
                     "max_violation": np.nan, "discharge_inversions": np.nan})
        return base

    x = np.asarray(res.x)
    veh = vehicles.set_index("veh_id")
    pos = positions.set_index("pos_id")

    loaded = set()
    total_length = 0.0
    lane_sequences = {}
    for q, (i,p_id,k) in enumerate(meta["x_vars"]):
        if x[q] > 0.5:
            loaded.add(i)
            total_length += float(veh.loc[i, "length"])
            lane_sequences.setdefault(p_id, []).append((k, int(veh.loc[i, "destination"]), i))

    inversions = 0
    for p_id, seq in lane_sequences.items():
        seq = sorted(seq)
        ds = [d for k,d,i in seq]
        inversions += sum(ds[a] > ds[a+1] for a in range(len(ds)-1))

    Ax = meta["A"].dot(x)
    upper = np.where(np.isfinite(meta["ub"]), np.maximum(Ax-meta["ub"],0),0)
    lower = np.where(np.isfinite(meta["lb"]), np.maximum(meta["lb"]-Ax,0),0)
    max_violation = float(max(np.max(upper), np.max(lower), 0.0))

    base.update({
        "objective": float(res.fun),
        "loaded": len(loaded),
        "deferred": n-len(loaded),
        "load_rate": len(loaded)/n,
        "length_utilization": total_length / float(positions.length_cap.sum()),
        "max_violation": max_violation,
        "discharge_inversions": inversions,
    })
    return base


def benchmark(out, ferry, sizes, seeds, congestions, time_limit):
    rows = []
    for congestion in congestions:
        for n in sizes:
            for seed in seeds:
                r = solve(n, seed, ferry, congestion, time_limit)
                rows.append(r)
                print(
                    f"{ferry:6s} {congestion:9s} n={n:3d} seed={seed:3d} "
                    f"t={r['runtime_s']:.2f}s load={r.get('loaded')} "
                    f"gap={r.get('mip_gap'):.4g} inv={r.get('discharge_inversions')} "
                    f"viol={r.get('max_violation')}"
                )
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ferry", choices=sorted(FERRY_PRESETS), default="small")
    ap.add_argument("--sizes", nargs="+", type=int, default=[30,50,70])
    ap.add_argument("--seeds", nargs="+", type=int, default=[11,22,33])
    ap.add_argument("--congestion", nargs="+", choices=["balanced","heavy","car_dense"],
                    default=["balanced"])
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--out", default="ordered_slot_full_mip.csv")
    args = ap.parse_args()
    df = benchmark(args.out, args.ferry, args.sizes, args.seeds, args.congestion, args.time_limit)
    print("\nSummary")
    print(df.groupby(["congestion","n"]).agg(
        runtime_mean=("runtime_s","mean"),
        loaded_mean=("loaded","mean"),
        load_rate_mean=("load_rate","mean"),
        util_mean=("length_utilization","mean"),
        gap_max=("mip_gap","max"),
        violation_max=("max_violation","max"),
        inversions_max=("discharge_inversions","max"),
    ).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
