
"""
Build a lightweight Full-MIP hardness predictor and solver-routing policy.

Hardness target (observable from a short diagnostic Full-MIP run):
- hard = 1 if the unrestricted Full-MIP either
  (a) does not certify mip_gap <= 1e-4 within the diagnostic time limit, or
  (b) consumes at least 90% of the diagnostic time budget.

Features are available before solving:
- problem size;
- fleet composition;
- aggregate length/mass statistics;
- compatibility density;
- average compatible positions/decks per vehicle;
- lane/deck capacity pressure proxies.

The model is intended only as a routing layer:
    easy -> Full-MIP
    hard -> adaptive NGFLO
"""

from pathlib import Path
import sys, importlib.util, time, json, math
import numpy as np
import pandas as pd
from scipy.optimize import milp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import joblib

ROOT = Path("/mnt/data")
OUT = ROOT / "ngflo_hardness_router_v1"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "base_mod", ROOT / "synthetic_ferry_ordered_full_mip.py"
)
base = importlib.util.module_from_spec(spec)
sys.modules["base_mod"] = base
spec.loader.exec_module(base)

DIAGNOSTIC_LIMIT = 1.0
GAP_THRESHOLD = 1e-4
RUNTIME_FRACTION_THRESHOLD = 0.90

REGIMES = ["balanced", "heavy", "car_dense"]
SIZES = [20, 30, 40, 50]
SEEDS = [11, 22, 33, 44]


def pre_solve_features(vehicles, positions, cfg, congestion):
    compat_counts = []
    compat_decks = []
    for _, v in vehicles.iterrows():
        poss = []
        decks = set()
        for _, p in positions.iterrows():
            if base.compatible(v, p, cfg):
                poss.append(int(p.pos_id))
                decks.add(int(p.deck))
        compat_counts.append(len(poss))
        compat_decks.append(len(decks))

    class_counts = vehicles["class"].value_counts(normalize=True).to_dict()
    total_lane_length = float(positions.length_cap.sum())
    total_lane_mass = float(positions.mass_cap.sum())
    total_vehicle_length = float(vehicles.length.sum())
    total_vehicle_mass = float(vehicles.mass.sum())

    return {
        "n": int(len(vehicles)),
        "congestion": congestion,
        "total_vehicle_length": total_vehicle_length,
        "mean_vehicle_length": float(vehicles.length.mean()),
        "std_vehicle_length": float(vehicles.length.std(ddof=0)),
        "max_vehicle_length": float(vehicles.length.max()),
        "total_vehicle_mass": total_vehicle_mass,
        "mean_vehicle_mass": float(vehicles.mass.mean()),
        "std_vehicle_mass": float(vehicles.mass.std(ddof=0)),
        "max_vehicle_mass": float(vehicles.mass.max()),
        "mean_vehicle_height": float(vehicles.height.mean()),
        "max_vehicle_height": float(vehicles.height.max()),
        "length_pressure": total_vehicle_length / total_lane_length,
        "mass_pressure": total_vehicle_mass / total_lane_mass,
        "mean_compatible_positions": float(np.mean(compat_counts)),
        "min_compatible_positions": int(np.min(compat_counts)),
        "mean_compatible_decks": float(np.mean(compat_decks)),
        "min_compatible_decks": int(np.min(compat_decks)),
        "compatibility_density": float(np.sum(compat_counts) / (len(vehicles) * len(positions))),
        "p_car": float(class_counts.get("car", 0.0)),
        "p_suv": float(class_counts.get("suv", 0.0)),
        "p_van": float(class_counts.get("van", 0.0)),
        "p_rigid_truck": float(class_counts.get("rigid_truck", 0.0)),
        "p_coach": float(class_counts.get("coach", 0.0)),
        "p_artic": float(class_counts.get("artic", 0.0)),
        "p_heavy": float(
            class_counts.get("rigid_truck", 0.0)
            + class_counts.get("coach", 0.0)
            + class_counts.get("artic", 0.0)
        ),
        "destination_entropy": float(
            -sum(
                p * math.log(max(p, 1e-12))
                for p in vehicles.destination.value_counts(normalize=True).values
            )
        ),
    }


def diagnostic_solve(vehicles, positions, cfg):
    c, integ, bounds, cons, meta = base.build_ordered_slot_model(
        vehicles, positions, cfg
    )
    t0 = time.perf_counter()
    res = milp(
        c=c, integrality=integ, bounds=bounds, constraints=cons,
        options={
            "time_limit": DIAGNOSTIC_LIMIT,
            "mip_rel_gap": 1e-6,
            "presolve": True,
        },
    )
    rt = time.perf_counter() - t0
    gap = float(getattr(res, "mip_gap", np.nan))
    solved = res.x is not None
    objective = float(res.fun) if solved and res.fun is not None else np.nan

    hard = (
        (not solved)
        or (not np.isfinite(gap))
        or (gap > GAP_THRESHOLD)
        or (rt >= RUNTIME_FRACTION_THRESHOLD * DIAGNOSTIC_LIMIT)
    )
    return {
        "diagnostic_runtime_s": rt,
        "diagnostic_mip_gap": gap,
        "diagnostic_objective": objective,
        "diagnostic_solved": bool(solved),
        "hard_label": int(hard),
        "n_binary": len(c),
        "mip_node_count": float(getattr(res, "mip_node_count", np.nan)),
    }


rows = []
cfg = base.FERRY_PRESETS["small"]
positions = base.make_positions(cfg)

for congestion in REGIMES:
    for n in SIZES:
        for seed in SEEDS:
            vehicles = base.generate_vehicles(n, seed, congestion)
            f = pre_solve_features(vehicles, positions, cfg, congestion)
            d = diagnostic_solve(vehicles, positions, cfg)
            row = {
                "instance_id": f"small_{congestion}_n{n}_s{seed}",
                "seed": seed,
                **f, **d
            }
            rows.append(row)
            print(
                f"{row['instance_id']:30s} hard={row['hard_label']} "
                f"t={row['diagnostic_runtime_s']:.3f}s "
                f"gap={row['diagnostic_mip_gap']}"
            )

df = pd.DataFrame(rows)
df.to_csv(OUT / "hardness_dataset.csv", index=False)

# Hold out seed 33 to mirror the prior test convention.
train = df[df.seed != 33].copy()
test = df[df.seed == 33].copy()

num_features = [
    "n","total_vehicle_length","mean_vehicle_length","std_vehicle_length",
    "max_vehicle_length","total_vehicle_mass","mean_vehicle_mass",
    "std_vehicle_mass","max_vehicle_mass","mean_vehicle_height","max_vehicle_height",
    "length_pressure","mass_pressure","mean_compatible_positions",
    "min_compatible_positions","mean_compatible_decks","min_compatible_decks",
    "compatibility_density","p_car","p_suv","p_van","p_rigid_truck",
    "p_coach","p_artic","p_heavy","destination_entropy"
]
cat_features = ["congestion"]

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])),
    ("clf", LogisticRegression(
        class_weight="balanced", max_iter=3000, C=0.7
    ))
])

pipe.fit(train[num_features + cat_features], train.hard_label)

for part in [train, test]:
    part["hard_probability"] = pipe.predict_proba(
        part[num_features + cat_features]
    )[:,1]
    # Routing threshold favors recall of hard states.
    part["predicted_hard"] = (part.hard_probability >= 0.40).astype(int)

def metrics(part):
    y = part.hard_label.to_numpy()
    p = part.predicted_hard.to_numpy()
    prob = part.hard_probability.to_numpy()
    return {
        "n": int(len(part)),
        "hard_prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y,p)),
        "balanced_accuracy": float(balanced_accuracy_score(y,p)),
        "precision_hard": float(precision_score(y,p,zero_division=0)),
        "recall_hard": float(recall_score(y,p,zero_division=0)),
        "f1_hard": float(f1_score(y,p,zero_division=0)),
        "roc_auc": float(roc_auc_score(y,prob)) if len(np.unique(y)) > 1 else np.nan,
        "confusion_matrix": confusion_matrix(y,p).tolist(),
    }

m = {"train": metrics(train), "test": metrics(test)}
with (OUT/"hardness_metrics.json").open("w") as f:
    json.dump(m,f,indent=2)

test_cols = [
    "instance_id","congestion","n","seed","hard_label","hard_probability",
    "predicted_hard","diagnostic_runtime_s","diagnostic_mip_gap",
    "length_pressure","mass_pressure","p_heavy","compatibility_density"
]
test[test_cols].to_csv(OUT/"heldout_predictions.csv", index=False)

# Extract standardized logistic coefficients for interpretability.
pre = pipe.named_steps["pre"]
clf = pipe.named_steps["clf"]
feature_names = pre.get_feature_names_out()
coef = pd.DataFrame({
    "feature": feature_names,
    "coefficient": clf.coef_[0]
}).sort_values("coefficient", ascending=False)
coef.to_csv(OUT/"router_coefficients.csv", index=False)

joblib.dump(pipe, OUT/"hardness_router.joblib")

print("\nMETRICS")
print(json.dumps(m,indent=2))
print("\nHELD OUT")
print(test[test_cols].to_string(index=False))
print("\nTOP POSITIVE HARDNESS COEFFICIENTS")
print(coef.head(10).to_string(index=False))
