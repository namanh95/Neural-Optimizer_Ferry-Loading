
"""
NGFLO robust fallback pipeline.

Pipeline
--------
1. Use pre-solve hardness router.
2. Easy state -> Full-MIP.
3. Hard state -> Adaptive NGFLO, K=6 -> 8 -> 12.
4. If K=12 still defers vehicles or remains under search pressure:
   expand to full candidate set and solve the unrestricted MIP with an
   incumbent objective cutoff c^T x <= J_NGFLO.

Important implementation note
-----------------------------
SciPy's milp interface does not expose an incumbent x0 / MIP-start argument.
Therefore this implementation uses a mathematically safe incumbent *objective cutoff*
rather than claiming a true solver warm start. If a solver with MIP-start support
is later used, the NGFLO incumbent can additionally be passed directly as x0.
"""

from pathlib import Path
import sys, importlib.util, time, json, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import milp, Bounds, LinearConstraint
import joblib

ROOT=Path("/mnt/data")
OUT=ROOT/"ngflo_fallback_pipeline_v1"
OUT.mkdir(exist_ok=True)

# ---------- Load baseline ----------
spec=importlib.util.spec_from_file_location("base_mod",ROOT/"synthetic_ferry_ordered_full_mip.py")
base=importlib.util.module_from_spec(spec)
sys.modules["base_mod"]=base
spec.loader.exec_module(base)

# ---------- Load hardness router ----------
router=joblib.load(ROOT/"ngflo_hardness_router_v1"/"hardness_router.joblib")
ROUTER_THRESHOLD=0.40
num_features=[
    "n","total_vehicle_length","mean_vehicle_length","std_vehicle_length",
    "max_vehicle_length","total_vehicle_mass","mean_vehicle_mass",
    "std_vehicle_mass","max_vehicle_mass","mean_vehicle_height","max_vehicle_height",
    "length_pressure","mass_pressure","mean_compatible_positions",
    "min_compatible_positions","mean_compatible_decks","min_compatible_decks",
    "compatibility_density","p_car","p_suv","p_van","p_rigid_truck",
    "p_coach","p_artic","p_heavy","destination_entropy"
]
cat_features=["congestion"]

# ---------- Load graph scorer ----------
ckpt=torch.load(
    ROOT/"ngflo_graph_scorer_v2"/"bipartite_graph_scorer.pt",
    map_location="cpu", weights_only=False
)
classes=ckpt["classes"]; class_to_idx={c:i for i,c in enumerate(classes)}
vmu=np.asarray(ckpt["vmu"],dtype=np.float32); vsd=np.asarray(ckpt["vsd"],dtype=np.float32)
pmu=np.asarray(ckpt["pmu"],dtype=np.float32); psd=np.asarray(ckpt["psd"],dtype=np.float32)
emu=np.asarray(ckpt["emu"],dtype=np.float32); esd=np.asarray(ckpt["esd"],dtype=np.float32)
veh_num=["length","width","height","mass","destination","priority"]
pos_num=["deck","lane","length_cap","width_cap","height_cap","mass_cap","y_coord"]
edge_num=["length_ratio","mass_ratio","abs_y_mass","destination_deck_distance"]

class BipartiteScorer(nn.Module):
    def __init__(self,v_in,p_in,e_in,h=32):
        super().__init__()
        self.venc=nn.Sequential(nn.Linear(v_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.penc=nn.Sequential(nn.Linear(p_in,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU())
        self.vupd=nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.pupd=nn.Sequential(nn.Linear(2*h,h),nn.ReLU())
        self.score=nn.Sequential(nn.Linear(4*h+e_in,64),nn.ReLU(),nn.Dropout(0.05),
                                 nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,d):
        vh=self.venc(d["vx"]); ph=self.penc(d["px"]); vi=d["vi"]; pi=d["pi"]
        vm=torch.zeros_like(vh); vc=torch.zeros((len(vh),1))
        pm=torch.zeros_like(ph); pc=torch.zeros((len(ph),1))
        ones=torch.ones((len(vi),1))
        vm.index_add_(0,vi,ph[pi]); vc.index_add_(0,vi,ones)
        pm.index_add_(0,pi,vh[vi]); pc.index_add_(0,pi,ones)
        vm=vm/vc.clamp_min(1); pm=pm/pc.clamp_min(1)
        vh2=self.vupd(torch.cat([vh,vm],1)); ph2=self.pupd(torch.cat([ph,pm],1))
        glob=torch.cat([vh2.mean(0),ph2.mean(0)]).unsqueeze(0).expand(len(vi),-1)
        z=torch.cat([vh2[vi],ph2[pi],d["ex"],glob],1)
        return self.score(z).squeeze(1)

model=BipartiteScorer(len(vmu)+len(classes),len(pmu),len(emu),32)
model.load_state_dict(ckpt["state_dict"]); model.eval(); torch.set_num_threads(1)

def hardness_features(vehicles,positions,cfg,congestion):
    compat_counts=[]; compat_decks=[]
    for _,v in vehicles.iterrows():
        poss=[]; decks=set()
        for _,p in positions.iterrows():
            if base.compatible(v,p,cfg):
                poss.append(int(p.pos_id)); decks.add(int(p.deck))
        compat_counts.append(len(poss)); compat_decks.append(len(decks))
    cc=vehicles["class"].value_counts(normalize=True).to_dict()
    probs=vehicles.destination.value_counts(normalize=True).values
    tl=float(positions.length_cap.sum()); tm=float(positions.mass_cap.sum())
    row={
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
        "length_pressure":float(vehicles.length.sum()/tl),
        "mass_pressure":float(vehicles.mass.sum()/tm),
        "mean_compatible_positions":float(np.mean(compat_counts)),
        "min_compatible_positions":int(np.min(compat_counts)),
        "mean_compatible_decks":float(np.mean(compat_decks)),
        "min_compatible_decks":int(np.min(compat_decks)),
        "compatibility_density":float(np.sum(compat_counts)/(len(vehicles)*len(positions))),
        "p_car":float(cc.get("car",0)),"p_suv":float(cc.get("suv",0)),
        "p_van":float(cc.get("van",0)),"p_rigid_truck":float(cc.get("rigid_truck",0)),
        "p_coach":float(cc.get("coach",0)),"p_artic":float(cc.get("artic",0)),
        "p_heavy":float(cc.get("rigid_truck",0)+cc.get("coach",0)+cc.get("artic",0)),
        "destination_entropy":float(-sum(p*math.log(max(p,1e-12)) for p in probs)),
        "congestion":congestion,
    }
    X=pd.DataFrame([row])
    prob=float(router.predict_proba(X[num_features+cat_features])[:,1][0])
    return prob

def score_edges(vehicles,positions,cfg):
    V=vehicles.sort_values("veh_id").copy(); P=positions.sort_values("pos_id").copy()
    vmap={int(x):i for i,x in enumerate(V.veh_id)}
    pmap={int(x):i for i,x in enumerate(P.pos_id)}
    vn=(V[veh_num].to_numpy(np.float32)-vmu)/vsd
    one=np.zeros((len(V),len(classes)),np.float32)
    for q,c in enumerate(V["class"]): one[q,class_to_idx[c]]=1
    vx=np.concatenate([vn,one],1)
    px=(P[pos_num].to_numpy(np.float32)-pmu)/psd
    rows=[]
    for _,v in V.iterrows():
        for _,p in P.iterrows():
            if base.compatible(v,p,cfg):
                rows.append({
                    "veh_id":int(v.veh_id),"pos_id":int(p.pos_id),
                    "length_ratio":float(v.length/p.length_cap),
                    "mass_ratio":float(v.mass/p.mass_cap),
                    "abs_y_mass":float(abs(p.y_coord)*v.mass),
                    "destination_deck_distance":float(abs((int(v.destination)-1)-min(int(p.deck),2)))
                })
    E=pd.DataFrame(rows)
    ex=(E[edge_num].to_numpy(np.float32)-emu)/esd
    vi=np.array([vmap[int(x)] for x in E.veh_id],np.int64)
    pi=np.array([pmap[int(x)] for x in E.pos_id],np.int64)
    d={"vx":torch.tensor(vx),"px":torch.tensor(px),"ex":torch.tensor(ex),
       "vi":torch.tensor(vi),"pi":torch.tensor(pi)}
    with torch.no_grad():
        E["score"]=torch.sigmoid(model(d)).numpy()
    E["rank"]=E.groupby("veh_id")["score"].rank(method="first",ascending=False).astype(int)
    return E

def solve_model(vehicles,positions,cfg,time_limit=5.0,allowed=None,objective_cutoff=None):
    c,integ,bounds,cons,m=base.build_ordered_slot_model(vehicles,positions,cfg)
    lb=np.zeros(len(c)); ub=np.ones(len(c))
    if allowed is not None:
        for idx,(i,p,k) in enumerate(m["x_vars"]):
            if p not in allowed.get(int(i),set()): ub[idx]=0.0
    constraints=[cons]
    if objective_cutoff is not None and np.isfinite(objective_cutoff):
        # The incumbent itself satisfies this cutoff, so it cannot remove all known feasible solutions.
        constraints.append(LinearConstraint(c, -np.inf, float(objective_cutoff)+1e-9))

    t0=time.perf_counter()
    res=milp(c=c,integrality=integ,bounds=Bounds(lb,ub),constraints=constraints,
             options={"time_limit":time_limit,"mip_rel_gap":1e-6,"presolve":True})
    rt=time.perf_counter()-t0

    if res.x is None:
        return {"success":False,"runtime_s":rt,"objective":np.nan,"loaded":np.nan,
                "deferred":np.nan,"mip_gap":np.nan,"solution_x":None,"meta":m,"c":c}
    x=np.asarray(res.x); loaded=set()
    for idx,(i,p,k) in enumerate(m["x_vars"]):
        if x[idx]>0.5: loaded.add(int(i))
    return {
        "success":True,"runtime_s":rt,"objective":float(res.fun),
        "loaded":len(loaded),"deferred":len(vehicles)-len(loaded),
        "mip_gap":float(getattr(res,"mip_gap",np.nan)),
        "solution_x":x,"meta":m,"c":c
    }

def adaptive_ngflo(vehicles,positions,cfg,E):
    total_time=0.0; final=None; stages=[]
    for K in [6,8,12]:
        allowed={int(vid):set(g.nlargest(min(K,len(g)),"score").pos_id.astype(int))
                 for vid,g in E.groupby("veh_id")}
        r=solve_model(vehicles,positions,cfg,time_limit=2.0,allowed=allowed)
        total_time+=r["runtime_s"]
        r["K"]=K
        if r["success"]:
            rank_lookup=E.set_index(["veh_id","pos_id"])["rank"].to_dict()
            selected=[]
            x=r["solution_x"]; m=r["meta"]
            for idx,(i,p,k) in enumerate(m["x_vars"]):
                if x[idx]>0.5:
                    selected.append(int(rank_lookup[(int(i),int(p))]))
            r["boundary_rate"]=sum(rr>=K for rr in selected)/max(1,len(selected))
        else:
            r["boundary_rate"]=1.0
        stages.append({k:v for k,v in r.items() if k not in {"solution_x","meta","c"}})
        final=r
        trigger=(not r["success"]) or r["deferred"]>0 or r["boundary_rate"]>0.10
        if not trigger:
            break
    return final,total_time,stages

def run_instance(congestion,n=50,seed=33):
    cfg=base.FERRY_PRESETS["small"]
    vehicles=base.generate_vehicles(n,seed,congestion); positions=base.make_positions(cfg)
    ph=hardness_features(vehicles,positions,cfg,congestion)

    # Benchmark unrestricted Full-MIP under 5 sec.
    full=solve_model(vehicles,positions,cfg,time_limit=5.0)

    if ph < ROUTER_THRESHOLD:
        return {
            "congestion":congestion,"n":n,"seed":seed,"hard_probability":ph,
            "route":"Full-MIP","fallback_used":False,
            "final_runtime_s":full["runtime_s"],"final_objective":full["objective"],
            "final_loaded":full["loaded"],"final_deferred":full["deferred"],
            "final_mip_gap":full["mip_gap"],
            "full_reference_runtime_s":full["runtime_s"],
            "full_reference_objective":full["objective"],
            "full_reference_mip_gap":full["mip_gap"],
            "adaptive_runtime_s":0.0,"fallback_runtime_s":0.0,
            "incumbent_before_fallback":np.nan,
        }, []

    E=score_edges(vehicles,positions,cfg)
    adaptive,adapt_time,stages=adaptive_ngflo(vehicles,positions,cfg,E)

    fallback_needed=(
        (not adaptive["success"]) or adaptive["deferred"]>0 or adaptive.get("boundary_rate",1)>0.10
    )
    if not fallback_needed:
        final=adaptive; fb_time=0.0
        route="Adaptive-NGFLO"
    else:
        incumbent=adaptive["objective"] if adaptive["success"] else None
        fallback=solve_model(
            vehicles,positions,cfg,time_limit=5.0,
            allowed=None, objective_cutoff=incumbent
        )
        fb_time=fallback["runtime_s"]
        # Keep best feasible incumbent.
        if fallback["success"] and (not adaptive["success"] or fallback["objective"] <= adaptive["objective"]+1e-9):
            final=fallback
        else:
            final=adaptive
        route="Adaptive-NGFLO + Full-Search Fallback"

    return {
        "congestion":congestion,"n":n,"seed":seed,"hard_probability":ph,
        "route":route,"fallback_used":fallback_needed,
        "final_runtime_s":adapt_time+fb_time,
        "final_objective":final["objective"],"final_loaded":final["loaded"],
        "final_deferred":final["deferred"],"final_mip_gap":final["mip_gap"],
        "full_reference_runtime_s":full["runtime_s"],
        "full_reference_objective":full["objective"],
        "full_reference_mip_gap":full["mip_gap"],
        "adaptive_runtime_s":adapt_time,"fallback_runtime_s":fb_time,
        "incumbent_before_fallback":adaptive["objective"] if adaptive["success"] else np.nan,
        "objective_change_vs_full_pct":100*(final["objective"]-full["objective"])/abs(full["objective"]),
        "runtime_change_vs_full_pct":100*(adapt_time+fb_time-full["runtime_s"])/full["runtime_s"],
    }, stages

rows=[]; stage_rows=[]
for congestion in ["balanced","heavy","car_dense"]:
    r,st=run_instance(congestion)
    rows.append(r)
    for s in st:
        s["congestion"]=congestion
        stage_rows.append(s)
    print(r)

pd.DataFrame(rows).to_csv(OUT/"fallback_pipeline_results.csv",index=False)
pd.DataFrame(stage_rows).to_csv(OUT/"adaptive_stage_results.csv",index=False)
print("\nFINAL")
print(pd.DataFrame(rows).to_string(index=False))
