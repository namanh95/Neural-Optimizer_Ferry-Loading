
"""
Deployable pre-solve solver router for the synthetic NGFLO benchmark.

Usage concept:
    route = recommend_solver(vehicles, positions, cfg, congestion)

Returns:
    "Full-MIP" or "Adaptive-NGFLO"

This router uses only pre-solve instance features. It does not run the MIP.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = Path("/mnt/data/ngflo_hardness_router_v1/hardness_router.joblib")
THRESHOLD = 0.40

NUM_FEATURES = [
    "n","total_vehicle_length","mean_vehicle_length","std_vehicle_length",
    "max_vehicle_length","total_vehicle_mass","mean_vehicle_mass",
    "std_vehicle_mass","max_vehicle_mass","mean_vehicle_height","max_vehicle_height",
    "length_pressure","mass_pressure","mean_compatible_positions",
    "min_compatible_positions","mean_compatible_decks","min_compatible_decks",
    "compatibility_density","p_car","p_suv","p_van","p_rigid_truck",
    "p_coach","p_artic","p_heavy","destination_entropy"
]
CAT_FEATURES = ["congestion"]

def extract_features(vehicles, positions, cfg, congestion, compatible_fn):
    compat_counts=[]; compat_decks=[]
    for _,v in vehicles.iterrows():
        poss=[]; decks=set()
        for _,p in positions.iterrows():
            if compatible_fn(v,p,cfg):
                poss.append(int(p.pos_id)); decks.add(int(p.deck))
        compat_counts.append(len(poss)); compat_decks.append(len(decks))

    cc=vehicles["class"].value_counts(normalize=True).to_dict()
    total_lane_length=float(positions.length_cap.sum())
    total_lane_mass=float(positions.mass_cap.sum())
    probs=vehicles.destination.value_counts(normalize=True).values

    return pd.DataFrame([{
        "n":len(vehicles),
        "total_vehicle_length":float(vehicles.length.sum()),
        "mean_vehicle_length":float(vehicles.length.mean()),
        "std_vehicle_length":float(vehicles.length.std(ddof=0)),
        "max_vehicle_length":float(vehicles.length.max()),
        "total_vehicle_mass":float(vehicles.mass.sum()),
        "mean_vehicle_mass":float(vehicles.mass.mean()),
        "std_vehicle_mass":float(vehicles.mass.std(ddof=0)),
        "max_vehicle_mass":float(vehicles.mass.max()),
        "mean_vehicle_height":float(vehicles.height.mean()),
        "max_vehicle_height":float(vehicles.height.max()),
        "length_pressure":float(vehicles.length.sum()/total_lane_length),
        "mass_pressure":float(vehicles.mass.sum()/total_lane_mass),
        "mean_compatible_positions":float(np.mean(compat_counts)),
        "min_compatible_positions":int(np.min(compat_counts)),
        "mean_compatible_decks":float(np.mean(compat_decks)),
        "min_compatible_decks":int(np.min(compat_decks)),
        "compatibility_density":float(np.sum(compat_counts)/(len(vehicles)*len(positions))),
        "p_car":float(cc.get("car",0)),
        "p_suv":float(cc.get("suv",0)),
        "p_van":float(cc.get("van",0)),
        "p_rigid_truck":float(cc.get("rigid_truck",0)),
        "p_coach":float(cc.get("coach",0)),
        "p_artic":float(cc.get("artic",0)),
        "p_heavy":float(cc.get("rigid_truck",0)+cc.get("coach",0)+cc.get("artic",0)),
        "destination_entropy":float(-sum(p*math.log(max(p,1e-12)) for p in probs)),
        "congestion":congestion,
    }])

def recommend_solver(vehicles, positions, cfg, congestion, compatible_fn, threshold=THRESHOLD):
    model=joblib.load(MODEL_PATH)
    X=extract_features(vehicles,positions,cfg,congestion,compatible_fn)
    prob=float(model.predict_proba(X[NUM_FEATURES+CAT_FEATURES])[:,1][0])
    route="Adaptive-NGFLO" if prob>=threshold else "Full-MIP"
    return {"route":route,"hard_probability":prob,"threshold":threshold}
