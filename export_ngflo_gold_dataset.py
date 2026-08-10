
"""
Export certified Full-MIP solutions into an NGFLO supervised dataset.

Outputs:
- instances.csv              : instance-level metadata and solver certificates
- vehicles.csv               : vehicle-node features
- positions.csv              : lane/deck-node features
- compatibility_edges.csv    : compatible vehicle-position edges
- assignment_labels.csv      : optimal vehicle-position labels (lane/deck level)
- slot_assignments.csv       : exact vehicle-position-slot assignments from Full-MIP
- dataset.jsonl              : compact per-instance graph records
- splits.csv                 : deterministic train/validation/test assignment

Gold-label criterion:
    MIP gap <= 1e-6
    max linear constraint violation <= 1e-8
    zero discharge-order inversions
"""

from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp

import synthetic_ferry_ordered_full_mip as base


def solve_and_export(instance_id, n, seed, ferry, congestion, time_limit=15.0):
    cfg = base.FERRY_PRESETS[ferry]
    vehicles = base.generate_vehicles(n, seed, congestion)
    positions = base.make_positions(cfg)
    c, integrality, bounds, constraints, meta = base.build_ordered_slot_model(
        vehicles, positions, cfg
    )

    t0 = time.perf_counter()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-8,
            "presolve": True,
        },
    )
    runtime = time.perf_counter() - t0

    if res.x is None:
        return None

    x = np.asarray(res.x)
    veh = vehicles.set_index("veh_id")
    pos = positions.set_index("pos_id")

    # Constraint residual certification.
    Ax = meta["A"].dot(x)
    upper = np.where(np.isfinite(meta["ub"]), np.maximum(Ax - meta["ub"], 0.0), 0.0)
    lower = np.where(np.isfinite(meta["lb"]), np.maximum(meta["lb"] - Ax, 0.0), 0.0)
    max_violation = float(max(np.max(upper), np.max(lower), 0.0))

    assignments = []
    lane_sequences = {}
    loaded = set()
    for q, (i, p_id, k) in enumerate(meta["x_vars"]):
        if x[q] > 0.5:
            loaded.add(i)
            assignments.append({
                "instance_id": instance_id,
                "veh_id": int(i),
                "pos_id": int(p_id),
                "slot": int(k),
                "deck": int(pos.loc[p_id, "deck"]),
                "lane": int(pos.loc[p_id, "lane"]),
                "destination": int(veh.loc[i, "destination"]),
            })
            lane_sequences.setdefault(int(p_id), []).append(
                (int(k), int(veh.loc[i, "destination"]), int(i))
            )

    inversions = 0
    for seq in lane_sequences.values():
        seq = sorted(seq)
        ds = [d for _, d, _ in seq]
        inversions += sum(ds[k] > ds[k+1] for k in range(len(ds)-1))

    mip_gap = float(getattr(res, "mip_gap", np.nan))
    gold = (
        np.isfinite(mip_gap)
        and mip_gap <= 1e-6
        and max_violation <= 1e-8
        and inversions == 0
    )

    # Vehicle records.
    vehicle_rows = []
    for _, v in vehicles.iterrows():
        vehicle_rows.append({
            "instance_id": instance_id,
            "veh_id": int(v.veh_id),
            "class": str(v["class"]),
            "length": float(v.length),
            "width": float(v.width),
            "height": float(v.height),
            "mass": float(v.mass),
            "destination": int(v.destination),
            "priority": int(v.priority),
            "loaded_label": int(v.veh_id in loaded),
        })

    # Position records.
    position_rows = []
    for _, p in positions.iterrows():
        position_rows.append({
            "instance_id": instance_id,
            "pos_id": int(p.pos_id),
            "deck": int(p.deck),
            "lane": int(p.lane),
            "length_cap": float(p.length_cap),
            "width_cap": float(p.width_cap),
            "height_cap": float(p.height_cap),
            "mass_cap": float(p.mass_cap),
            "y_coord": float(p.y_coord),
        })

    # Lane/deck compatibility edges and labels.
    opt_pos = {int(a["veh_id"]): int(a["pos_id"]) for a in assignments}
    edge_rows = []
    label_rows = []
    for _, v in vehicles.iterrows():
        for _, p in positions.iterrows():
            if base.compatible(v, p, cfg):
                i = int(v.veh_id)
                p_id = int(p.pos_id)
                length_ratio = float(v.length / p.length_cap)
                mass_ratio = float(v.mass / p.mass_cap)
                edge_rows.append({
                    "instance_id": instance_id,
                    "veh_id": i,
                    "pos_id": p_id,
                    "length_ratio": length_ratio,
                    "mass_ratio": mass_ratio,
                    "abs_y_mass": float(abs(p.y_coord) * v.mass),
                    "destination_deck_distance": float(
                        abs((int(v.destination)-1) - min(int(p.deck), 2))
                    ),
                })
                label_rows.append({
                    "instance_id": instance_id,
                    "veh_id": i,
                    "pos_id": p_id,
                    "label": int(opt_pos.get(i, -1) == p_id),
                })

    instance_row = {
        "instance_id": instance_id,
        "ferry": ferry,
        "congestion": congestion,
        "n": int(n),
        "seed": int(seed),
        "objective": float(res.fun),
        "runtime_s": float(runtime),
        "mip_gap": mip_gap,
        "mip_node_count": float(getattr(res, "mip_node_count", np.nan)),
        "max_violation": max_violation,
        "discharge_inversions": int(inversions),
        "loaded": int(len(loaded)),
        "deferred": int(n - len(loaded)),
        "gold_label": bool(gold),
    }

    json_record = {
        "instance": instance_row,
        "vehicles": vehicle_rows,
        "positions": position_rows,
        "edges": edge_rows,
        "positive_assignments": assignments,
    }

    return {
        "instance": instance_row,
        "vehicles": vehicle_rows,
        "positions": position_rows,
        "edges": edge_rows,
        "labels": label_rows,
        "assignments": assignments,
        "json": json_record,
    }


def deterministic_split(instance_df):
    # Stratified-ish deterministic split by sorted regime/size/seed.
    rows = []
    for (cong, n), grp in instance_df.groupby(["congestion", "n"], sort=True):
        grp = grp.sort_values("seed").reset_index(drop=True)
        m = len(grp)
        for idx, r in grp.iterrows():
            if m >= 3:
                split = ["train", "validation", "test"][idx % 3]
            elif m == 2:
                split = ["train", "test"][idx]
            else:
                split = "train"
            rows.append({"instance_id": r.instance_id, "split": split})
    return pd.DataFrame(rows)


def main(out_dir="ngflo_gold_dataset_v1"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # First trusted dataset: all three regimes, n <= 40, three seeds.
    regimes = ["balanced", "heavy", "car_dense"]
    sizes = [20, 30]
    seeds = [11, 22, 33]

    instances, vehicles, positions, edges, labels, assignments, json_records = (
        [], [], [], [], [], [], []
    )

    counter = 0
    for congestion in regimes:
        for n in sizes:
            for seed in seeds:
                instance_id = f"small_{congestion}_n{n}_s{seed}"
                rec = solve_and_export(
                    instance_id=instance_id,
                    n=n,
                    seed=seed,
                    ferry="small",
                    congestion=congestion,
                    time_limit=15.0,
                )
                if rec is None:
                    print("FAILED", instance_id)
                    continue
                print(
                    f"{instance_id:30s} gap={rec['instance']['mip_gap']:.3g} "
                    f"gold={rec['instance']['gold_label']} "
                    f"loaded={rec['instance']['loaded']} "
                    f"t={rec['instance']['runtime_s']:.2f}s"
                )
                if rec["instance"]["gold_label"]:
                    instances.append(rec["instance"])
                    vehicles.extend(rec["vehicles"])
                    positions.extend(rec["positions"])
                    edges.extend(rec["edges"])
                    labels.extend(rec["labels"])
                    assignments.extend(rec["assignments"])
                    json_records.append(rec["json"])
                counter += 1

    instance_df = pd.DataFrame(instances)
    vehicle_df = pd.DataFrame(vehicles)
    position_df = pd.DataFrame(positions)
    edge_df = pd.DataFrame(edges)
    label_df = pd.DataFrame(labels)
    assignment_df = pd.DataFrame(assignments)

    split_df = deterministic_split(instance_df)
    instance_df = instance_df.merge(split_df, on="instance_id", how="left")

    instance_df.to_csv(out / "instances.csv", index=False)
    vehicle_df.to_csv(out / "vehicles.csv", index=False)
    position_df.to_csv(out / "positions.csv", index=False)
    edge_df.to_csv(out / "compatibility_edges.csv", index=False)
    label_df.to_csv(out / "assignment_labels.csv", index=False)
    assignment_df.to_csv(out / "slot_assignments.csv", index=False)
    split_df.to_csv(out / "splits.csv", index=False)

    with (out / "dataset.jsonl").open("w", encoding="utf-8") as f:
        split_map = dict(zip(split_df.instance_id, split_df.split))
        for rec in json_records:
            rec["instance"]["split"] = split_map[rec["instance"]["instance_id"]]
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # Diagnostics.
    positives = int(label_df["label"].sum()) if len(label_df) else 0
    total_edges = len(label_df)
    diagnostics = {
        "instances": int(len(instance_df)),
        "vehicles": int(len(vehicle_df)),
        "positions": int(len(position_df)),
        "compatibility_edges": int(len(edge_df)),
        "positive_assignment_edges": positives,
        "positive_edge_rate": float(positives / total_edges) if total_edges else np.nan,
        "exact_slot_assignments": int(len(assignment_df)),
        "train_instances": int((instance_df["split"] == "train").sum()),
        "validation_instances": int((instance_df["split"] == "validation").sum()),
        "test_instances": int((instance_df["split"] == "test").sum()),
        "max_mip_gap": float(instance_df["mip_gap"].max()),
        "max_constraint_violation": float(instance_df["max_violation"].max()),
        "max_discharge_inversions": int(instance_df["discharge_inversions"].max()),
    }
    with (out / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nDiagnostics")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/data/ngflo_gold_dataset_v1")
    args = ap.parse_args()
    main(args.out_dir)
